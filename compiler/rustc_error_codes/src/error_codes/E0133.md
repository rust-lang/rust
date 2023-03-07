Unsafe code was used outside of an unsafe function or block.

Erroneous code example:

```compile_fail,E0133
unsafe fn f() { return; } // This is the unsafe code

fn main() {
    f(); // error: call to unsafe function requires unsafe function or block
}
```

Using unsafe functionality is potentially dangerous and disallowed by safety
checks. Examples:

* Dereferencing raw pointers
* Calling functions via FFI
* Calling functions marked unsafe

These safety checks can be relaxed for a section of the code by wrapping the
unsafe instructions with an `unsafe` block. For instance:

```
unsafe fn f() { return; }

fn main() {
    unsafe { f(); } // ok!
}
```

See the [unsafe section][unsafe-section] of the Book for more details.

[unsafe-section]: https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
