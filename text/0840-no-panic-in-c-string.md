- Feature Name: non_panicky_cstring
- Start Date: 2015-02-13
- RFC PR: https://github.com/rust-lang/rfcs/pull/840
- Rust Issue: https://github.com/rust-lang/rust/issues/22470

# Summary

Remove panics from `CString::from_slice` and `CString::from_vec`, making
these functions return `Result` instead.

# Motivation

> As I shivered and brooded on the casting of that brain-blasting shadow,
> I knew that I had at last pried out one of earth’s supreme horrors—one of
> those nameless blights of outer voids whose faint daemon scratchings we
> sometimes hear on the farthest rim of space, yet from which our own finite
> vision has given us a merciful immunity.
>
> — H. P. Lovecraft, <cite>The Lurking Fear</cite>

Currently the functions that produce `std::ffi::CString` out of Rust byte
strings panic when the input contains NUL bytes. As strings containing NULs
are not commonly seen in real-world usage, it is easy for developers to
overlook the potential panic unless they test for such atypical input.

The panic is particularly sneaky when hidden behind an API using regular Rust
string types. Consider this example:

```rust
fn set_text(text: &str) {
    let c_text = CString::from_slice(text.as_bytes());  // panic lurks here
    unsafe { ffi::set_text(c_text.as_ptr()) };
}
```

This implementation effectively imposes a requirement on the input string to
contain no inner NUL bytes, which is generally permitted in pure Rust.
This restriction is not apparent in the signature of the function and needs to
be described in the documentation. Furthermore, the creator of the code may be
oblivious to the potential panic.

The conventions on failure modes elsewhere in Rust libraries tend to limit
panics to outcomes of programmer errors. Functions validating external data
should return `Result` to allow graceful handling of the errors.

# Detailed design

The return types of `CString::from_slice` and `CString::from_vec` is changed
to `Result`:

```rust
impl CString {
    pub fn from_slice(s: &[u8]) -> Result<CString, NulError> { ... }
    pub fn from_vec(v: Vec<u8>) -> Result<CString, IntoCStrError> { ... }
}
```

The error type `NulError` provides information on the position of the first
NUL byte found in the string. `IntoCStrError` wraps `NulError` and also
provides the `Vec` which has been moved into `CString::from_vec`.

`std::error::FromError` implementations are provided to convert the error
types above to `std::io::Error` of the `InvalidInput` kind. This facilitates
use of the conversion functions in input-processing code.

# Proof-of-concept implementation

The proposed changes are implemented in a crates.io project
[c_string](https://github.com/mzabaluev/rust-c-str), where the analog of
`CString` is named `CStrBuf`.

# Drawbacks

The need to extract the data from a `Result` in the success case is annoying.
However, it may be viewed as a speed bump to make the developer aware of a
potential failure and to require an explicit choice on how to handle it.
Even the least graceful way, a call to `unwrap`, makes the potential panic
apparent in the code.

# Alternatives

Non-panicky functions can be added alongside the existing functions, e.g.,
as `from_slice_failing`. Adding new functions complicates the API where little
reason for that exists; composition is preferred to adding function variants.
Longer function names, together with a less convenient return value, may deter
people from using the safer functions.

The panicky functions could also be renamed to `unpack_slice` and `unpack_vec`,
respectively, to highlight their conceptual proximity to `unpack`.

If the panicky behavior is preserved, plentiful possibilities for DoS attacks
and other unforeseen failures in the field may be introduced by code oblivious
to the input constraints.

# Unresolved questions

None.
