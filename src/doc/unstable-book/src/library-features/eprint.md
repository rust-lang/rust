# `eprint`

The tracking issue for this feature is: [#40528]

[#40528]: https://github.com/rust-lang/rust/issues/40528

------------------------

This feature enables the `eprint!` and `eprintln!` global macros,
which are just like `print!` and `println!`, respectively, except that
they send output to the standard _error_ stream, rather than standard
output.  (`panic!` messages have always been written to standard error.)

