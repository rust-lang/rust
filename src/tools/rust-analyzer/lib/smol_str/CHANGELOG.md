# Changelog

## 0.3.1 - 2024-09-04

- Fix `SmolStrBuilder` leaking implementation details

## 0.3.0 - 2024-09-04

- Removed deprecated `SmolStr::new_inline_from_ascii` function
- Removed `SmolStr::to_string` in favor of `ToString::to_string`
- Added `impl AsRef<[u8]> for SmolStr` impl
- Added `impl AsRef<OsStr> for SmolStr` impl
- Added `impl AsRef<Path> for SmolStr` impl
- Added `SmolStrBuilder`
