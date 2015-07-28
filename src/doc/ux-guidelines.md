% User Experience Guidelines

These guidelines are for the rustc compiler.

## Error Messages

* The word "illegal" is illegal. Prefer "invalid" or a more specific word
instead.
* Errors and Warnings should not suggest how to fix the problem. A Help
message should be emitted instead.
* Errors, Warnings, Notes, and Help messages start with a lowercase letter and
do not end with punctuation.

## Error Explanations

* Prefer 'the compiler' to 'Rust' or 'rustc'. See
[E0004](https://doc.rust-lang.org/stable/error-index.html#E0004) for an
example.
