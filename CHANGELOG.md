# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1](https://github.com/francisdb/wavefront-obj-io/compare/v0.3.0...v0.3.1) - 2026-09-19

### Other

- write floats with the Zmij shortest formatter
- format floats straight into the line buffer

## [0.3.0](https://github.com/francisdb/wavefront-obj-io/compare/v0.2.0...v0.3.0) - 2026-08-25

### Other

- *(deps)* bump actions/checkout from 6 to 7 ([#5](https://github.com/francisdb/wavefront-obj-io/pull/5))
- *(deps)* bump Swatinem/rust-cache from 2.9.1 to 2.9.2 ([#7](https://github.com/francisdb/wavefront-obj-io/pull/7))
- [**breaking**] replace itoa with std NumBuffer ([#8](https://github.com/francisdb/wavefront-obj-io/pull/8))

## [0.2.0](https://github.com/francisdb/wavefront-obj-io/compare/v0.1.1...v0.2.0) - 2026-05-06

### Added

- add MTL reader/writer trait pair ([#3](https://github.com/francisdb/wavefront-obj-io/pull/3))

### Other

- *(deps)* [**breaking**] bump itoa to 1.0.18

## [0.1.1](https://github.com/francisdb/wavefront-obj-io/compare/v0.1.0...v0.1.1) - 2026-05-05

### Other

- stop excluding test fixtures from the package
- move fixture-based tests to tests/round_trip.rs
- add release-plz workflow
