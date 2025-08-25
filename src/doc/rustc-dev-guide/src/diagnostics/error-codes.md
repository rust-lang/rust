# Error codes
We generally try to assign each error message a unique code like `E0123`. These
codes are defined in the compiler in the `diagnostics.rs` files found in each
crate, which basically consist of macros. All error codes have an associated
explanation: new error codes must include them. Note that not all _historical_
(no longer emitted) error codes have explanations.

## Error explanations

The explanations are written in Markdown (see the [CommonMark Spec] for
specifics around syntax), and all of them are linked in the [`rustc_error_codes`]
crate. Please read [RFC 1567] for details on how to format and write long error
codes. As of <!-- date-check --> February 2023, there is an
effort[^new-explanations] to replace this largely outdated RFC with a new more
flexible standard.

Error explanations should expand on the error message and provide details about
_why_ the error occurs. It is not helpful for users to copy-paste a quick fix;
explanations should help users understand why their code cannot be accepted by
the compiler. Rust prides itself on helpful error messages and long-form
explanations are no exception. However, before error explanations are
overhauled[^new-explanations] it is a bit open as to how exactly they should be
written, as always: ask your reviewer or ask around on the Rust Zulip.

[^new-explanations]: See the draft RFC [here][new-explanations-rfc].

[`rustc_error_codes`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_error_codes/index.html
[CommonMark Spec]: https://spec.commonmark.org/current/
[RFC 1567]: https://github.com/rust-lang/rfcs/blob/master/text/1567-long-error-codes-explanation-normalization.md
[new-explanations-rfc]: https://github.com/rust-lang/rfcs/pull/3370

## Allocating a fresh code

Error codes are stored in `compiler/rustc_error_codes`.

To create a new error, you first need to find the next available
code. You can find it with `tidy`:

```
./x test tidy
```

This will invoke the tidy script, which generally checks that your code obeys
our coding conventions. Some of these jobs check error codes and ensure that
there aren't duplicates, etc (the tidy check is defined in
`src/tools/tidy/src/error_codes.rs`). Once it is finished with that, tidy will
print out the highest used error code:

```
...
tidy check
Found 505 error codes
Highest error code: `E0591`
...
```

Here we see the highest error code in use is `E0591`, so we _probably_ want
`E0592`. To be sure, run `rg E0592` and check, you should see no references.

You will have to write an extended description for your error,
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

To actually issue the error, you can use the `struct_span_code_err!` macro:

```rust
struct_span_code_err!(self.dcx(), // some path to the `DiagCtxt` here
                 span, // whatever span in the source you want
                 E0592, // your new error code
                 fluent::example::an_error_message)
    .emit() // actually issue the error
```

If you want to add notes or other snippets, you can invoke methods before you
call `.emit()`:

```rust
struct_span_code_err!(...)
    .span_label(another_span, fluent::example::example_label)
    .span_note(another_span, fluent::example::separate_note)
    .emit()
```

For an example of a PR adding an error code, see [#76143].

[#76143]: https://github.com/rust-lang/rust/pull/76143

## Running error code doctests

To test the examples added in `rustc_error_codes/src/error_codes`, run the
error index generator using:

```
./x test ./src/tools/error_index_generator
```
