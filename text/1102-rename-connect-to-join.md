- Feature Name: `rename_connect_to_join`
- Start Date: 2015-05-02
- RFC PR: [rust-lang/rfcs#1102](https://github.com/rust-lang/rfcs/pull/1102)
- Rust Issue: [rust-lang/rust#26900](https://github.com/rust-lang/rust/issues/26900)

# Summary

Rename `.connect()` to `.join()` in `SliceConcatExt`.

# Motivation

Rust has a string concatenation method named `.connect()` in `SliceConcatExt`.
However, this does not align with the precedents in other languages. Most
languages use `.join()` for that purpose, as seen later.

This is probably because, in the ancient Rust, `join` was a keyword to join a
task. However, `join` retired as a keyword in 2011 with the commit
rust-lang/rust@d1857d3. While `.connect()` is technically correct, the name may
not be directly inferred by the users of the mainstream languages. There was [a
question] about this on reddit.

[a question]: http://www.reddit.com/r/rust/comments/336rj3/whats_the_best_way_to_join_strings_with_a_space/

The languages that use the name of `join` are:

- Python: [str.join](https://docs.python.org/3/library/stdtypes.html#str.join)
- Ruby: [Array.join](http://ruby-doc.org/core-2.2.0/Array.html#method-i-join)
- JavaScript: [Array.prototype.join](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/join)
- Go: [strings.Join](https://golang.org/pkg/strings/#Join)
- C#: [String.Join](https://msdn.microsoft.com/en-us/library/dd783876%28v=vs.110%29.aspx?f=255&MSPPError=-2147217396)
- Java: [String.join](http://docs.oracle.com/javase/8/docs/api/java/lang/String.html#join-java.lang.CharSequence-java.lang.Iterable-)
- Perl: [join](http://perldoc.perl.org/functions/join.html)

The languages not using `join` are as follows. Interestingly, they are
all functional-ish languages.

- Haskell: [intercalate](http://hackage.haskell.org/package/text-1.2.0.4/docs/Data-Text.html#v:intercalate)
- OCaml: [String.concat](http://caml.inria.fr/pub/docs/manual-ocaml/libref/String.html#VALconcat)
- F#: [String.concat](https://msdn.microsoft.com/en-us/library/ee353761.aspx)

Note that Rust also has `.concat()` in `SliceConcatExt`, which is a specialized
version of `.connect()` that uses an empty string as a separator.

Another reason is that the term "join" already has similar usage in the standard
library. There are `std::path::Path::join` and `std::env::join_paths` which are
used to join the paths.

# Detailed design

While the `SliceConcatExt` trait is unstable, the `.connect()` method itself is
marked as stable. So we need to:

1. Deprecate the `.connect()` method.
2. Add the `.join()` method.

Or, if we are to achieve the [instability guarantee], we may remove the old
method entirely, as it's still pre-1.0. However, the author considers that this
may require even more consensus.

[instability guarantee]: https://github.com/rust-lang/rust/issues/24928

# Drawbacks

Having a deprecated method in a newborn language is not pretty.

If we do remove the `.connect()` method, the language becomes pretty again, but
it breaks the stability guarantee at the same time.

# Alternatives

Keep the status quo. Improving searchability in the docs will help newcomers
find the appropriate method.

# Unresolved questions

Are there even more clever names for the method? How about `.homura()`, or
`.madoka()`?
