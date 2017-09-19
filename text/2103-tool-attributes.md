- Feature Name: tool_attributes, tool_lints
- Start Date: 2016-09-22
- RFC PR: https://github.com/rust-lang/rfcs/pull/2103
- Rust Issue: https://github.com/rust-lang/rust/issues/44690


# Summary
[summary]: #summary

This RFC proposes a temporary solution to the problem of letting tools use
attributes. We outline a (partial) long-term solution and propose a step towards
that solution for tools which are part of the Rust distribution.

The long-term solution is that a crate can use attributes for a specific tool by
using some explicit (but unspecified) opt-in mechanism. The tool name then
becomes the root of a module hierarchy for attribute naming. E.g., by opting-in
to a tool named `my_tool`, a crate can use `#[my_tool::foo]` and
`#[my_tool::bar(42)]`, etc.

This RFC is a special case of the long-term solution: any tool distributed with
Rust creates a scope for attributes (without any opt-in). So any crate can use
`#[rustdoc::hidden]` or `#[rustfmt::skip]`.

E.g.,

```
#[rustfmt::skip]
fn foo() {}
```

This would be allowed by the compiler but ignored. When Rustfmt is run on the
crate, it will read the attibute and skip formatting `foo` (note that we make no
provision for reading the attribute or doing anything with it, that is all up to
the tool).

This RFC proposes a second mechanism for scoping lints for tools. Similar to
attributes, we propose a subset of a hypothetical long-term solution.

This RFC supersedes #1755.

# Motivation
[motivation]: #motivation

Attributes are a useful, general-purpose mechanism for annotating code with
metadata. They are used in the language (e.g., `repr`), for macros (e.g.,
`derive`, and for user-supplied attribute- like macros), and by tools
(e.g., `rustfmt_skip` which instructs Rustfmt not to format an item).
Attributes could also be used by compiler plugins such as lints.

Currently, custom attributes (i.e., those not known to the compiler, e.g.,
`rustfmt_skip`) are unstable. There is a future compatibility hazard with custom
attributes: if we add `#[foo]` to the language, then any users using a `foo`
custom attribute will suffer breakage.

There is a potential problem with the interaction between custom attributes and
attribute-like macros. Given an attribute, the compiler cannot tell if the
attribute is intended to be a macro invocation or an attribute that might only
be used by a tool (either outside or inside the compiler). Currently, the
compiler tries to find a macro and if it cannot, ignores the attribute (giving a
stability error if not on nightly or the `custom_attribute` feature is not
enabled). However, if the user intended the attribute to be a macro, silently
ignoring the missing macro error is not the right thing to do. The compiler
needs to know whether an attribute is intended to be a macro or not.

Given the above constraints, an opt-in solution is attractive. However, any such
solution ends up being closely related to mechanisms for importing crates
(`extern crate`) and macro naming. These features are being re-examined or
are unstable and so now is a bad time to fully specify a long-term solution.

We do wish to make progress on allowing tools to use attributes. For example,
Rustfmt is mostly ready to move towards stabilisation, but requires some kind of
`skip` attribute. So we are proposing a solution that should work well with any
reasonable long-term solution and addresses the needs of some important tools
today.

Similarly, tools (e.g., Clippy) may want to use their own lints without the
compiler warning about unused lints. E.g., we want a user to be able to write
`#![allow(clippy::some_lint)]` in their crate without warning.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

### Attributes

This section assumes that attributes (e.g., `#[test]`) have already been taught.

You can use attibutes in your crate to pass information to tools. For now, this
facility is limited to the tools we include with the Rust distribution.

The names of these attributes are a path starting with the name of a tool, and
then one or more identifiers, e.g., `#[tool_name::foo]` or
`#[tool_name::bar::baz::qux(argument)]`. Such paths hide any attribute-like
macros with the same name and location.

For example, using `#[rustfmt::skip]` indicates that an item (such as a function)
should not be formatted by Rustfmt:

```
#[rustfmt::skip]
fn foo() { this_will_be_kept_as_is_by_rustfmt(); }

fn bar() { this_will_be_reformatted }

mod baz {
    #![rustfmt::skip]
    // Rustfmt will skip this whole module.
}
```

### Lints

This section assumes lints have already been taught.

Lints can be defined hierarchically as a path, as well as just a single name.
For example, `nonstandard_style::non_snake_case_functions` and
`nonstandard_style::uppercase_variables`. Note this RFC is not proposing
changing any existing lints, just extending the current lint naming system. Lint
names cannot be imported using `use`.

Lints can be enforced by tools other than the compiler. For example, Clippy
provides a large suite of lints to catch common mistakes and improve your Rust
code. Lints for tools are prefixed with the tool name, e.g., `clippy::box_vec`.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Long-term solution

There will be some opt-in mechanism for crates to declare that they want to
allow use of a tool's attributes. This might be in the source text (an attribute
as in #1755 or new syntax, e.g., `extern attribute foo;`) or passed to rustc as
a command line flag (e.g., `--extern-attr foo`). The exact mechanism is
deliberately unspecifed.

After opting-in to `foo`, a crate can use `foo` as the base of a path in any
attribute in the crate. E.g., allowing `#[foo::bar]` to be used (but not
`#[foo]`). This mechanism is follows the normal macro hygiene rules. Depending
on the opt-in mechanism a tool might be able to specify to the compiler which
paths are valid, e.g., allow `#[foo::bar]` but disallow `#[foo::baz]`. I would
hope that we'd be able to reuse most of the macro naming feature (see #1561)
here (i.e., this won't be a whole new specification, we'll just allow a new way
to base paths).

Unscoped attributes will be reserved for the language and can't be used by tools.

During macro expansion, when faced with an attribute, the compiler first tries
to find a macro using the [macro name resolution rules](https://github.com/rust-lang/rfcs/blob/master/text/1561-macro-naming.md).
The compiler then checks if the attribute matches any of the declared or built-
in attributes. If this fails, then it reports a macro not found error. The
compiler *may* suggest mis-typed attributes (declared or built-in).

A similar opt-in mechanism will exist for lints.


## Proposed for immediate implementation

There is an attribute path white list of the names of tools shipped with the Rust
distribution. Any crate can use an attibute path starting with those names and
the attribute will not trigger the custom attribute lint or require a macro
feature gate.

E.g., `#[rustdoc::foo]` will be permitted in stable Rust code; `#[rustdoc]` will
still be treated as a custom attribute.

The initial list of allowed prefixes is `rustc`, `rustdoc`, and `rls` (but see
note below on activation). As tools are added to the distribution, they will be
allowed as path prefixes in attributes. We expect to add `rustfmt` and `clippy`
in the near future. Note that whether one of these names can be used does not
depend on whether the relevant component is installed on the user's system; this
is a simple, universal white list.

Given the earlier rules on name resolution, these attributes would shadow any
attribute macro with the same name. This is not problematic because a macro
would have to be in a module starting with a tool name (e.g., `rustdoc::foo`),
naming macros in such a way is currently unstable, and this can be worked around
by using an import (`use`).

Tool-scoped attributes should be preserved by the compiler for as long as
possible through compilation. This allows tools which plug into the compiler
(like Clippy) to observe these attributes on items during type checking, etc.

Likewise, white-listed tools may be used as a prefix for lints. So for example,
`rustfmt::foo` and `clippy::bar` are both valid lint names, from the compiler's
perspective.


### Activation and unused attibutes/lints

For each name on the whitelist, it is indicated if the name is active for
attributes or lints. A name is only activated if required. So for example,
`rustdoc` will not be activated at all until it takes advantage of this feature.
I expect `clippy` will be activated only for lints and attributes, and `rustfmt`
only for attributes.

A tool that has an active name *must* check for unused lints/attibutes. For
example, if `rustfmt` becomes active for attributes, and only recognises
`rustfmt::skip`, it must produce a warning if a user uses `rustfmt::foo` in
their code.

These two requirements together mean that we do not lose checking of unused
attributes/lints in any circumstance and we can move to having the compiler
check for unused attributes/lints as part of a possible long-term solution
without introducing new warnings or errors.


### Forward and backward compatability

Since custom attributes are feature gated and scoped attributes are part of the
unstable macros 2.0 work, there is no backwards compatability issue.

For tools who want to move to these newly stable attributes (e.g., from
`rustfmt_skip` to `rustfmt::skip`) they will have to manage the change
themselves.

Although the mechanism for opt-in for the long-term solution is unspecified, the
actual usage of tool attributes seems pretty clear. Therefore we can be reasonably
confident that this proposal is forward-compatible in its syntax, etc.

For the white-listed tools, will their names be implicitly imported in the long-
term solution? One could imagine either leaving them implicit (similar to the
libraries prelude) or using warning cycles or an epoch to move them to explicit
opt-in.


# Drawbacks
[drawbacks]: #drawbacks

The proposed scheme does not allow tools or macros to use custom top-level
attributes (I consider this a feature, not a bug, but others may differ).

Some tools are clearly given special treatment.

We permit some useless attributes without warning from the compiler (e.g.,
`#[rustfmt::foo]`, assuming Rustfmt does nothing with `foo`). However, tools
should warn or error on such attributes.

We are not planning any infrastructure to help tools use these attributes. That
seems fine for now, I imagine a long-term solution should include some library
or API for this.

No interaction with imports or other parts of the module system.

# Alternatives
[alternatives]: #alternatives

We could continue to force tools to rely on `cfg_attr` - this is very
unergonomic, e.g., `#[cfg_attr(rustfmt, rustfmt_skip)]`.

We could allow all scoped attributes without checks. This feels like it
introduces too much scope for error.

# Unresolved questions
[unresolved]: #unresolved-questions

Are there other tools that should be included on the whitelist (`#[test]` perhaps)?

Should we try and move some top-level attributes that are compiler-specific
(rather than language-specific) to use `#[rustc::]`? (E.g., `crate_type`).

How should the compiler expose path lints to lint plugins/lint tools?

[RFC 2126](https://github.com/rust-lang/rfcs/blob/master/text/2126-path-clarity.md)
may change how paths are written, the paths used in attributes in this RFC should
be adjusted accordingly.
