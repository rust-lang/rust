# Apple notification group

**Github Labels:** [O-macos], [O-ios], [O-tvos], [O-watchos] and [O-visionos] <br>
**Ping command:** `@rustbot ping apple`

This list will be used to ask for help both in diagnosing and testing
Apple-related issues as well as suggestions on how to resolve interesting
questions regarding our macOS/iOS/tvOS/watchOS/visionOS support.

To get a better idea for what the group will do, here are some examples of the
kinds of questions where we would have reached out to the group for advice in
determining the best course of action:

* Raising the minimum supported versions (e.g. [#104385])
* Additional Apple targets (e.g. [#121419])
* Obscure Xcode linker details (e.g. [#121430])

[O-macos]: https://github.com/rust-lang/rust/labels/O-macos
[O-ios]: https://github.com/rust-lang/rust/labels/O-ios
[O-tvos]: https://github.com/rust-lang/rust/labels/O-tvos
[O-watchos]: https://github.com/rust-lang/rust/labels/O-watchos
[O-visionos]: https://github.com/rust-lang/rust/labels/O-visionos
[#104385]: https://github.com/rust-lang/rust/pull/104385
[#121419]: https://github.com/rust-lang/rust/pull/121419
[#121430]: https://github.com/rust-lang/rust/pull/121430

## Deployment targets

Apple platforms have a concept of "deployment target", controlled with the
`*_DEPLOYMENT_TARGET` environment variables, and specifies the minimum OS
version that a binary runs on.

Using an API from a newer OS version in the standard library than the default
that `rustc` uses will result in either a static or a dynamic linker error.
For this reason, try to suggest that people document on `extern "C"` APIs
which OS version they were introduced with, and if that's newer than the
current default used by `rustc`, suggest to use weak linking.

## The App Store and private APIs

Apple are very protective about using undocumented APIs, so it's important
that whenever a change uses a new function, that they are verified to actually
be public API, as even just mentioning undocumented APIs in the binary
(without calling it) can lead to rejections from the App Store.

For example, Darwin / the XNU kernel actually has futex syscalls, but we can't
use them in `std` because they are not public API.

In general, for an API to be considered public by Apple, it has to:
- Appear in a public header (i.e. one distributed with Xcode, and found for
  the specific platform under `xcrun --show-sdk-path --sdk $SDK`).
- Have an availability attribute on it (like `__API_AVAILABLE`,
  `API_AVAILABLE` or similar).
