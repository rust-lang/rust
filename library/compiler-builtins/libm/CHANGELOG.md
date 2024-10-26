# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.9](https://github.com/rust-lang/libm/compare/libm-v0.2.8...libm-v0.2.9) - 2024-10-26

### Fixed

- Update exponent calculations in nextafter to match musl

### Changed

- Update licensing to MIT AND (MIT OR Apache-2.0), as this is derivative from
  MIT-licensed musl.
- Set edition to 2021 for all crates
- Upgrade all dependencies

### Other

- Don't deny warnings in lib.rs
- Rename the `musl-bitwise-tests` feature to `test-musl-serialized`
- Rename the `musl-reference-tests` feature to `musl-bitwise-tests`
- Move `musl-reference-tests` to a new `libm-test` crate
- Add a `force-soft-floats` feature to prevent using any intrinsics or
  arch-specific code
- Deny warnings in CI
- Fix `clippy::deprecated_cfg_attr` on compiler_builtins
- Corrected English typos
- Remove unneeded `extern core` in `tgamma`
- Allow internal_features lint when building with "unstable"

## [v0.2.1] - 2019-11-22

### Fixed

- sincosf

## [v0.2.0] - 2019-10-18

### Added

- Benchmarks
- signum
- remainder
- remainderf
- nextafter
- nextafterf

### Fixed

- Rounding to negative zero
- Overflows in rem_pio2 and remquo
- Overflows in fma
- sincosf

### Removed

- F32Ext and F64Ext traits

## [v0.1.4] - 2019-06-12

### Fixed

- Restored compatibility with Rust 1.31.0

## [v0.1.3] - 2019-05-14

### Added

- minf
- fmin
- fmaxf
- fmax

## [v0.1.2] - 2018-07-18

### Added

- acosf
- asin
- asinf
- atan
- atan2
- atan2f
- atanf
- cos
- cosf
- cosh
- coshf
- exp2
- expm1
- expm1f
- expo2
- fmaf
- pow
- sin
- sinf
- sinh
- sinhf
- tan
- tanf
- tanh
- tanhf

## [v0.1.1] - 2018-07-14

### Added

- acos
- acosf
- asin
- asinf
- atanf
- cbrt
- cbrtf
- ceil
- ceilf
- cosf
- exp
- exp2
- exp2f
- expm1
- expm1f
- fdim
- fdimf
- floorf
- fma
- fmod
- log
- log2
- log10
- log10f
- log1p
- log1pf
- log2f
- roundf
- sinf
- tanf

## v0.1.0 - 2018-07-13

- Initial release

[Unreleased]: https://github.com/japaric/libm/compare/v0.2.1...HEAD
[v0.2.1]: https://github.com/japaric/libm/compare/0.2.0...v0.2.1
[v0.2.0]: https://github.com/japaric/libm/compare/0.1.4...v0.2.0
[v0.1.4]: https://github.com/japaric/libm/compare/0.1.3...v0.1.4
[v0.1.3]: https://github.com/japaric/libm/compare/v0.1.2...0.1.3
[v0.1.2]: https://github.com/japaric/libm/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/japaric/libm/compare/v0.1.0...v0.1.1
