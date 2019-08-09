# `non_ascii_idents`

The tracking issue for this feature is: [#55467]

[#55467]: https://github.com/rust-lang/rust/issues/55467

------------------------

The `non_ascii_idents` feature adds support for non-ASCII identifiers.

## Examples

```rust
#![feature(non_ascii_idents)]

const ε: f64 = 0.00001f64;
const Π: f64 = 3.14f64;
```

## Changes to the language reference

> **<sup>Lexer:<sup>**  
> IDENTIFIER :  
> &nbsp;&nbsp; &nbsp;&nbsp; XID_start XID_continue<sup>\*</sup>  
> &nbsp;&nbsp; | `_` XID_continue<sup>+</sup>  

An identifier is any nonempty Unicode string of the following form:

Either

   * The first character has property [`XID_start`]
   * The remaining characters have property [`XID_continue`]

Or

   * The first character is `_`
   * The identifier is more than one character, `_` alone is not an identifier
   * The remaining characters have property [`XID_continue`]

that does _not_ occur in the set of [strict keywords].

> **Note**: [`XID_start`] and [`XID_continue`] as character properties cover the
> character ranges used to form the more familiar C and Java language-family
> identifiers.

[`XID_start`]:  http://unicode.org/cldr/utility/list-unicodeset.jsp?a=%5B%3AXID_Start%3A%5D&abb=on&g=&i=
[`XID_continue`]: http://unicode.org/cldr/utility/list-unicodeset.jsp?a=%5B%3AXID_Continue%3A%5D&abb=on&g=&i=
[strict keywords]: ../../reference/keywords.md#strict-keywords
