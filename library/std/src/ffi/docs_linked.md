## On Unix

On Unix, [`OsStr`] implements the
<code>std::os::unix::ffi::[OsStrExt][unix.OsStrExt]</code> trait, which
augments it with two methods, [`from_bytes`] and [`as_bytes`].
These do inexpensive conversions from and to byte slices.

Additionally, on Unix [`OsString`] implements the
<code>std::os::unix::ffi::[OsStringExt][unix.OsStringExt]</code> trait,
which provides [`from_vec`] and [`into_vec`] methods that consume
their arguments, and take or produce vectors of [`u8`].

## On Windows

An [`OsStr`] can be losslessly converted to a native Windows string. And
a native Windows string can be losslessly converted to an [`OsString`].

On Windows, [`OsStr`] implements the
<code>std::os::windows::ffi::[OsStrExt][windows.OsStrExt]</code> trait,
which provides an [`encode_wide`] method. This provides an
iterator that can be [`collect`]ed into a vector of [`u16`]. After a nul
characters is appended, this is the same as a native Windows string.

Additionally, on Windows [`OsString`] implements the
<code>std::os::windows:ffi::[OsStringExt][windows.OsStringExt]</code>
trait, which provides a [`from_wide`] method to convert a native Windows
string (without the terminating nul character) to an [`OsString`].

[unix.OsStringExt]: crate::os::unix::ffi::OsStringExt "os::unix::ffi::OsStringExt"
[`from_vec`]: crate::os::unix::ffi::OsStringExt::from_vec "os::unix::ffi::OsStringExt::from_vec"
[`into_vec`]: crate::os::unix::ffi::OsStringExt::into_vec "os::unix::ffi::OsStringExt::into_vec"
[unix.OsStrExt]: crate::os::unix::ffi::OsStrExt "os::unix::ffi::OsStrExt"
[`from_bytes`]: crate::os::unix::ffi::OsStrExt::from_bytes "os::unix::ffi::OsStrExt::from_bytes"
[`as_bytes`]: crate::os::unix::ffi::OsStrExt::as_bytes "os::unix::ffi::OsStrExt::as_bytes"
[windows.OsStrExt]: crate::os::windows::ffi::OsStrExt "os::windows::ffi::OsStrExt"
[`encode_wide`]: crate::os::windows::ffi::OsStrExt::encode_wide "os::windows::ffi::OsStrExt::encode_wide"
[`collect`]: crate::iter::Iterator::collect "iter::Iterator::collect"
[windows.OsStringExt]: crate::os::windows::ffi::OsStringExt "os::windows::ffi::OsStringExt"
[`from_wide`]: crate::os::windows::ffi::OsStringExt::from_wide "os::windows::ffi::OsStringExt::from_wide"
