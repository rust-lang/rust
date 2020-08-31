# Diagnostic Codes
We generally try to assign each error message a unique code like `E0123`. These
codes are defined in the compiler in the `diagnostics.rs` files found in each
crate, which basically consist of macros. The codes come in two varieties: those
that have an extended write-up, and those that do not. Whenever possible, if you
are making a new code, you should write an extended write-up.

### Allocating a fresh code

Error codes are stored in `compiler/rustc_error_codes`.

To create a new error, you first need to find the next available
code. You can find it with `tidy`:

```
./x.py test tidy
```

This will invoke the tidy script, which generally checks that your code obeys
our coding conventions. One of those jobs is to check that diagnostic codes are
indeed unique. Once it is finished with that, tidy will print out the lowest
unused code:

```
...
tidy check (x86_64-apple-darwin)
* 470 error codes
* highest error code: E0591
...
```

Here we see the highest error code in use is `E0591`, so we _probably_ want
`E0592`. To be sure, run `rg E0592` and check, you should see no references.

Ideally, you will write an extended description for your error,
which will go in `rustc_error_codes/src/error_codes/E0592.md`.
To register the error, open `rustc_error_codes/src/error_codes.rs` and add the
code (in its proper numerical order) into` register_diagnostics!` macro, like
this:

```rust
register_diagnostics! {
    ...
    E0592: include_str!("./error_codes/E0592.md"),
}
```

But you can also add it without an extended description:

```rust
register_diagnostics! {
    ...
    E0592, // put a description here
}
```

To actually issue the error, you can use the `struct_span_err!` macro:

```rust
struct_span_err!(self.tcx.sess, // some path to the session here
                 span, // whatever span in the source you want
                 E0592, // your new error code
                 &format!("text of the error"))
    .emit() // actually issue the error
```

If you want to add notes or other snippets, you can invoke methods before you
call `.emit()`:

```rust
struct_span_err!(...)
    .span_label(another_span, "something to label in the source")
    .span_note(another_span, "some separate note, probably avoid these")
    .emit_()
```

For an example of a PR adding an error code, see [#76143].

[#76143]: https://github.com/rust-lang/rust/pull/76143
