- Start Date: 2014-09-16
- RFC PR #: https://github.com/rust-lang/rfcs/pull/234
- Rust Issue #: https://github.com/rust-lang/rust/issues/17323

# Summary

Make enum variants part of both the type and value namespaces.

# Motivation

We might, post-1.0, want to allow using enum variants as types. This would be
backwards incompatible, because if a module already has a value with the same name
as the variant in scope, then there will be a name clash.

# Detailed design

Enum variants would always be part of both the type and value namespaces.
Variants would not, however, be usable as types - we might want to allow this
later, but it is out of scope for this RFC.

## Data

Occurrences of name clashes in the Rust repo:

* `Key` in `rustrt::local_data`

* `InAddr` in `native::io::net`

* `Ast` in `regex::parse`

* `Class` in `regex::parse`

* `Native` in `regex::re`

* `Dynamic` in `regex::re`

* `Zero` in `num::bigint`

* `String` in `term::terminfo::parm`

* `String` in `serialize::json`

* `List` in `serialize::json`

* `Object` in `serialize::json`

* `Argument` in `fmt_macros`

* `Metadata` in `rustc_llvm`

* `ObjectFile` in `rustc_llvm`

* 'ItemDecorator' in `syntax::ext::base`

* 'ItemModifier' in `syntax::ext::base`

* `FunctionDebugContext` in `rustc::middle::trans::debuginfo`

* `AutoDerefRef` in `rustc::middle::ty`

* `MethodParam` in `rustc::middle::typeck`

* `MethodObject` in `rustc::middle::typeck`

That's a total of 20 in the compiler and libraries.


# Drawbacks

Prevents the common-ish idiom of having a struct with the same name as a variant
and then having a value of that struct be the variant's data.

# Alternatives

Don't do it. That would prevent us making changes to the typed-ness of enums in
the future. If we accept this RFC, but at some point we decide we never want to
do anything with enum variants and types, we could always roll back this change
backwards compatibly.

# Unresolved questions

N/A
