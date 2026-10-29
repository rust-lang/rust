//@compile-flags: -Zmiri-strict-provenance

fn empty() -> &'static str {
    ""
}

fn hello() -> &'static str {
    "Hello, world!"
}

fn hello_bytes() -> &'static [u8; 13] {
    b"Hello, world!"
}

fn hello_bytes_fat() -> &'static [u8] {
    b"Hello, world!"
}

fn fat_pointer_on_32_bit() {
    Some(5).expect("foo");
}

fn str_indexing() {
    let mut x = "Hello".to_string();
    let _v = &mut x[..3]; // Test IndexMut on String.
}

fn unique_aliasing() {
    // This is a regression test for the aliasing rules of a `Unique<T>` pointer.
    // At the time of writing this test case, Miri does not treat `Unique<T>`
    // pointers as a special case, these are treated like any other raw pointer.
    // However, there are existing GitHub issues which may lead to `Unique<T>`
    // becoming a special case through asserting unique ownership over the pointee:
    // - https://github.com/rust-lang/unsafe-code-guidelines/issues/258
    // - https://github.com/rust-lang/unsafe-code-guidelines/issues/262
    // Below, the calls to `String::remove` and `String::insert[_str]` follow
    // code paths that would trigger undefined behavior in case `Unique<T>`
    // would ever assert semantic ownership over the pointee. Internally,
    // these methods call `self.vec.as_ptr()` and `self.vec.as_mut_ptr()` on
    // the vector of bytes that are backing the `String`. That `Vec<u8>` holds a
    // `Unique<u8>` internally. The second call to `Vec::as_mut_ptr(&mut self)`
    // would then invalidate the pointers derived from `Vec::as_ptr(&self)`.
    // Note that as long as `Unique<T>` is treated like any other raw pointer,
    // this test case should pass. It is merely here as a canary test for
    // potential future undefined behavior.
    let mut x = String::from("Hello");
    assert_eq!(x.remove(0), 'H');
    x.insert(0, 'H');
    assert_eq!(x, "Hello");
    x.insert_str(x.len(), ", world!");
    assert_eq!(x, "Hello, world!");
}

fn main() {
    assert_eq!(empty(), "");
    assert_eq!(hello(), "Hello, world!");
    assert_eq!(hello_bytes(), b"Hello, world!");
    assert_eq!(hello_bytes_fat(), b"Hello, world!");

    fat_pointer_on_32_bit(); // Should run without crashing.
    str_indexing();
    unique_aliasing();
}
