## On Unix

On Unix, [`OsStr`] implements the
<code>std::os::unix::ffi::OsStrExt</code> trait, which
augments it with two methods, `from_bytes` and `as_bytes`.
These do inexpensive conversions from and to byte slices.

Additionally, on Unix [`OsString`] implements the
<code>std::os::unix::ffi::OsStringExt</code> trait,
which provides `from_vec` and `into_vec` methods that consume
their arguments, and take or produce vectors of [`u8`].

## On Windows

An [`OsStr`] can be losslessly converted to a native Windows string. And
a native Windows string can be losslessly converted to an [`OsString`].

On Windows, [`OsStr`] implements the
<code>std::os::windows::ffi::OsStrExt</code> trait,
which provides an `encode_wide` method. This provides an
iterator that can be [`collect`]ed into a vector of [`u16`]. After a nul
characters is appended, this is the same as a native Windows string.

Additionally, on Windows [`OsString`] implements the
<code>std::os::windows:ffi::OsStringExt</code>
trait, which provides a `from_wide` method to convert a native Windows
string (without the terminating nul character) to an [`OsString`].

[`collect`]: crate::iter::Iterator::collect "iter::Iterator::collect"
