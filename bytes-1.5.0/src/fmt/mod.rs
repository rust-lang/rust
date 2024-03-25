mod debug;
mod hex;

/// `BytesRef` is not a part of public API of bytes crate.
struct BytesRef<'a>(&'a [u8]);
