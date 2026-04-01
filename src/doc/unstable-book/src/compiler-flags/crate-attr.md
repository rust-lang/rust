# `crate-attr`

The tracking issue for this feature is: [#138287](https://github.com/rust-lang/rust/issues/138287).

------------------------

The `-Z crate-attr` flag allows you to inject attributes into the crate root.
For example, `-Z crate-attr=crate_name="test"` acts as if `#![crate_name="test"]` were present before the first source line of the crate root.

To inject multiple attributes, pass `-Z crate-attr` multiple times.

Formally, the expansion behaves as follows:
1. The crate is parsed as if `-Z crate-attr` were not present.
2. The attributes in `-Z crate-attr` are parsed.
3. The attributes are injected at the top of the crate root.
4. Macro expansion is performed.
