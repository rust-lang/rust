- Feature Name: procedural_macros
- Start Date: 2016-02-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/1566
- Rust Issue: https://github.com/rust-lang/rust/issues/38356

# Summary
[summary]: #summary

This RFC proposes an evolution of Rust's procedural macro system (aka syntax
extensions, aka compiler plugins). This RFC specifies syntax for the definition
of procedural macros, a high-level view of their implementation in the compiler,
and outlines how they interact with the compilation process.

This RFC specifies the architecture of the procedural macro system. It relies on
[RFC 1561](https://github.com/rust-lang/rfcs/pull/1561) which specifies the
naming and modularisation of macros. It leaves many of the details for further
RFCs, in particular the details of the APIs available to macro authors
(tentatively called `libproc_macro`, formerly `libmacro`). See this
[blog post](http://ncameron.org/blog/libmacro/) for some ideas of how that might
look.

[RFC 1681](https://github.com/rust-lang/rfcs/pull/1681) specified a mechanism
for custom derive using 'macros 1.1'. That RFC is essentially a subset of this
one. Changes and differences are noted throughout the text.

At the highest level, macros are defined by implementing functions marked with
a `#[proc_macro]` attribute. Macros operate on a list of tokens provided by the
compiler and return a list of tokens that the macro use is replaced by. We
provide low-level facilities for operating on these tokens. Higher level
facilities (e.g., for parsing tokens to an AST) should exist as library crates.


# Motivation
[motivation]: #motivation

Procedural macros have long been a part of Rust and have been used for diverse
and interesting purposes, for example [compile-time regexes](https://github.com/rust-lang-nursery/regex),
[serialisation](https://github.com/serde-rs/serde), and
[design by contract](https://github.com/nrc/libhoare). They allow the ultimate
flexibility in syntactic abstraction, and offer possibilities for efficiently
using Rust in novel ways.

Procedural macros are currently unstable and are awkward to define. We would
like to remedy this by implementing a new, simpler system for procedural macros,
and for this new system to be on the usual path to stabilisation.

One major problem with the current system is that since it is based on ASTs, if
we change the Rust language (even in a backwards compatible way) we can easily
break procedural macros. Therefore, offering the usual backwards compatibility
guarantees to procedural macros, would inhibit our ability to evolve the
language. By switching to a token-based (rather than AST- based) system, we hope
to avoid this problem.

# Detailed design
[design]: #detailed-design

There are two kinds of procedural macro: function-like and attribute-like. These
two kinds exist today, and other than naming (see
[RFC 1561](https://github.com/rust-lang/rfcs/pull/1561)) the syntax for using
these macros remains unchanged. If the macro is called `foo`, then a function-
like macro is used with syntax `foo!(...)`, and an attribute-like macro with
`#[foo(...)] ...`. Macros may be used in the same places as `macro_rules` macros
and this remains unchanged.

There is also a third kind, custom derive, which are specified in [RFC
1681](https://github.com/rust-lang/rfcs/pull/1681). This RFC extends the
facilities open to custom derive macros beyond the string-based system of RFC
1681.

To define a procedural macro, the programmer must write a function with a
specific signature and attribute. Where `foo` is the name of a function-like
macro:

```
#[proc_macro]
pub fn foo(TokenStream) -> TokenStream;
```

The first argument is the tokens between the delimiters in the macro use.
For example in `foo!(a, b, c)`, the first argument would be `[Ident(a), Comma,
Ident(b), Comma, Ident(c)]`.

The value returned replaces the macro use.

Attribute-like:

```
#[proc_macro_attribute]
pub fn foo(Option<TokenStream>, TokenStream) -> TokenStream;
```

The first argument is a list of the tokens between the delimiters in the macro
use. Examples:

* `#[foo]` => `None`
* `#[foo()]` => `Some([])`
* `#[foo(a, b, c)]` => `Some([Ident(a), Comma, Ident(b), Comma, Ident(c)])`

The second argument is the tokens for the AST node the attribute is placed on.
Note that in order to compute the tokens to pass here, the compiler must be able
to parse the code the attribute is applied to. However, the AST for the node
passed to the macro is discarded, it is not passed to the macro nor used by the
compiler (in practice, this might not be 100% true due to optimisiations). If
the macro wants an AST, it must parse the tokens itself.

The attribute and the AST node it is applied to are both replaced by the
returned tokens. In most cases, the tokens returned by a procedural macro will
be parsed by the compiler. It is the procedural macro's responsibility to ensure
that the tokens parse without error. In some cases, the tokens will be consumed
by another macro without parsing, in which case they do not need to parse. The
distinction is not statically enforced. It could be, but I don't think the
overhead would be justified.

Custom derive:

```
#[proc_macro_derive]
pub fn foo(TokenStream) -> TokenStream;
```

Similar to attribute-like macros, the item a custom derive applies to must
parse. Custom derives may on be applied to the items that a built-in derive may
be applied to (structs and enums).

Currently, macros implementing custom derive only have the option of converting
the `TokenStream` to a string and converting a result string back to a
`TokenStream`. This option will remain, but macro authors will also be able to
operate directly on the `TokenStream` (which should be preferred, since it
allows for hygiene and span support).

Procedural macros which take an identifier before the argument list (e.g, `foo!
bar(...)`) will not be supported (at least initially).

My feeling is that this macro form is not used enough to justify its existence.
From a design perspective, it encourages uses of macros for language extension,
rather than syntactic abstraction. I feel that such macros are at higher risk of
making programs incomprehensible and of fragmenting the ecosystem).

Behind the scenes, these functions implement traits for each macro kind. We may
in the future allow implementing these traits directly, rather than just
implementing the above functions. By adding methods to these traits, we can
allow macro implementations to pass data to the compiler, for example,
specifying hygiene information or allowing for fast re-compilation.

## `proc-macro` crates

[Macros 1.1](https://github.com/rust-lang/rfcs/pull/1681) added a new crate
type: proc-macro. This both allows procedural macros to be declared within the
crate, and dictates how the crate is compiled. Procedural macros must use
this crate type.

We introduce a special configuration option: `#[cfg(proc_macro)]`. Items with
this configuration are not macros themselves but are compiled only for macro
uses.

If a crate is a `proc-macro` crate, then the `proc_macro` cfg variable is true
for the whole crate. Initially it will be false for all other crates. This has
the effect of partitioning crates into macro- defining and non-macro defining
crates. In the future, I hope we can relax these restrictions so that macro and
non-macro code can live in the same crate.

Importing macros for use means using `extern crate` to make the crate available
and then using `use` imports or paths to name macros, just like other items.
Again, see [RFC 1561](https://github.com/rust-lang/rfcs/pull/1561) for more
details.

When a `proc-macro` crate is `extern crate`ed, it's items (even public ones) are
not available to the importing crate; only macros declared in that crate. There
should be a lint to warn about public items which will not be visible due to
`proc_macro`. The crate is used by the compiler at compile-time, rather than
linked with the importing crate at runtime.

[Macros 1.1](https://github.com/rust-lang/rfcs/pull/1681) required `#[macro_use]`
on `extern crate` which imports procedural macros. This will not be required
and should be deprecated.


## Writing procedural macros

Procedural macro authors should not use the compiler crates (libsyntax, etc.).
Using these will remain unstable. We will make available a new crate,
libproc_macro, which will follow the usual path to stabilisation, will be part
of the Rust distribution, and will be required to be used by procedural macros
(because, at the least, it defines the types used in the required signatures).

The details of libproc_macro will be specified in a future RFC. In the meantime,
this [blog post](http://ncameron.org/blog/libmacro/) gives an idea of what it
might contain.

The philosophy here is that libproc_macro will contain low-level tools for
constructing macros, dealing with tokens, hygiene, pattern matching, quasi-
quoting, interactions with the compiler, etc. For higher level abstractions
(such as parsing and an AST), macros should use external libraries (there are no
restrictions on `#[cfg(proc_macro)]` crates using other crates).

A `MacroContext` is an object placed in thread-local storage when a macro is
expanded. It contains data about how the macro is being used and defined. It is
expected that for most uses, macro authors will not use the `MacroContext`
directly, but it will be used by library functions. It will be more fully
defined in the upcoming RFC proposing libproc_macro.

Rust macros are hygienic by default. Hygiene is a large and complex subject, but
to summarise: effectively, naming takes place in the context of the macro
definition, not the expanded macro.

Procedural macros often want to bend the rules around macro hygiene, for example
to make items or variables more widely nameable than they would be by default.
Procedural macros will be able to take part in the application of the hygiene
algorithm via libproc_macro. Again, full details must wait for the libproc_macro
RFC and a sketch is available in this [blog post](http://ncameron.org/blog/libmacro/).


## Tokens

Procedural macros will primarily operate on tokens. There are two main benefits
to this principle: flexibility and future proofing. By operating on tokens, code
passed to procedural macros does not need to satisfy the Rust parser, only the
lexer. Stabilising an interface based on tokens means we need only commit to
not changing the rules around those tokens, not the whole grammar. I.e., it
allows us to change the Rust grammar without breaking procedural macros.

In order to make the token-based interface even more flexible and future-proof,
I propose a simpler token abstraction than is currently used in the compiler.
The proposed system may be used directly in the compiler or may be an interface
wrapper over a more efficient representation.

Since macro expansion will not operate purely on tokens, we must keep hygiene
information on tokens, rather than on `Ident` AST nodes (we might be able to
optimise by not keeping such info for all tokens, but that is an implementation
detail). We will also keep span information for each token, since that is where
a record of macro expansion is maintained (and it will make life easier for
tools. Again, we might optimise internally).

A token is a single lexical element, for example, a numeric literal, a word
(which could be an identifier or keyword), a string literal, or a comment.

A token stream is a sequence of tokens, e.g., `a b c;` is a stream of four
tokens - `['a', 'b', 'c', ';'']`.

A token tree is a tree structure where each leaf node is a token and each
interior node is a token stream. I.e., a token stream which can contain nested
token streams. A token tree can be delimited, e.g., `a (b c);` will give
`TT(None, ['a', TT(Some('()'), ['b', 'c'], ';'']))`. An undelimited token tree
is useful for grouping tokens due to expansion, without representation in the
source code. That could be used for unsafety hygiene, or to affect precedence
and parsing without affecting scoping. They also replace the interpolated AST
tokens currently in the compiler.

In code:

```
// We might optimise this representation
pub struct TokenStream(Vec<TokenTree>);

// A borrowed TokenStream
pub struct TokenSlice<'a>(&'a [TokenTree]);

// A token or token tree.
pub struct TokenTree {
    pub kind: TokenKind,
    pub span: Span,
    pub hygiene: HygieneObject,
}

pub enum TokenKind {
    Sequence(Delimiter, TokenStream),

    // The content of the comment can be found from the span.
    Comment(CommentKind),

    // `text` is the string contents, not including delimiters. It would be nice
    // to avoid an allocation in the common case that the string is in the
    // source code. We might be able to use `&'codemap str` or something.
    // `raw_markers` is for the count of `#`s if the string is a raw string. If
    // the string is not raw, then it will be `None`.
    String { text: Symbol, raw_markers: Option<usize>, kind: StringKind },

    // char literal, span includes the `'` delimiters.
    Char(char),

    // These tokens are treated specially since they are used for macro
    // expansion or delimiting items.
    Exclamation,  // `!`
    Dollar,       // `$`
    // Not actually sure if we need this or if semicolons can be treated like
    // other punctuation.
    Semicolon,    // `;`
    Eof,          // Do we need this?

    // Word is defined by Unicode Standard Annex 31 -
    // [Unicode Identifier and Pattern Syntax](http://unicode.org/reports/tr31/)
    Word(Symbol),
    Punctuation(char),
}

pub enum Delimiter {
    None,
    // { }
    Brace,
    // ( )
    Parenthesis,
    // [ ]
    Bracket,
}

pub enum CommentKind {
    Regular,
    InnerDoc,
    OuterDoc,
}

pub enum StringKind {
    Regular,
    Byte,
}

// A Symbol is a possibly-interned string.
pub struct Symbol { ... }
```

Note that although tokens exclude whitespace, by examining the spans of tokens,
a procedural macro can get the string representation of a `TokenStream` and thus
has access to whitespace information.

### Open question: `Punctuation(char)` and multi-char operators.

Rust has many compound operators, e.g., `<<`. It's not clear how best to deal
with them. If the source code contains "`+ =`", it would be nice to distinguish
this in the token stream from "`+=`". On the other hand, if we represent `<<` as
a single token, then the macro may need to split them into `<`, `<` in generic
position.

I had hoped to represent each character as a separate token. However, to make
pattern matching backwards compatible, we would need to combine some tokens. In
fact, if we want to be completely backwards compatible, we probably need to keep
the same set of compound operators as are defined at the moment.

Some solutions:

* `Punctuation(char)` with special rules for pattern matching tokens,
* `Punctuation([char])` with a facility for macros to split tokens. Tokenising
  could match the maximum number of punctuation characters, or use the rules for
  the current token set. The former would have issues with pattern matching. The
  latter is a bit hacky, there would be backwards compatibility issues if we
  wanted to add new compound operators in the future.

## Staging

1. Implement [RFC 1561](https://github.com/rust-lang/rfcs/pull/1561).
2. Implement `#[proc_macro]` and `#[cfg(proc_macro)]` and the function approach to
   defining macros. However, pass the existing data structures to the macros,
   rather than tokens and `MacroContext`.
3. Implement libproc_macro and make this available to macros. At this stage both old
   and new macros are available (functions with different signatures). This will
   require an RFC and considerable refactoring of the compiler.
4. Implement some high-level macro facilities in external crates on top of
   libproc_macro. It is hoped that much of this work will be community-led.
5. After some time to allow conversion, deprecate the old-style macros. Later,
   remove old macros completely.


# Drawbacks
[drawbacks]: #drawbacks

Procedural macros are a somewhat unpleasant corner of Rust at the moment. It is
hard to argue that some kind of reform is unnecessary. One could find fault with
this proposed reform in particular (see below for some alternatives). Some
drawbacks that come to mind:

* providing such a low-level API risks never seeing good high-level libraries;
* the design is complex and thus will take some time to implement and stabilise,
  meanwhile unstable procedural macros are a major pain point in current Rust;
* dealing with tokens and hygiene may discourage macro authors due to complexity,
  hopefully that is addressed by library crates.

The actual concept of procedural macros also have drawbacks: executing arbitrary
code in the compiler makes it vulnerable to crashes and possibly security issues,
macros can introduce hard to debug errors, macros can make a program hard to
comprehend, it risks creating de facto dialects of Rust and thus fragmentation
of the ecosystem, etc.

# Alternatives
[alternatives]: #alternatives

We could keep the existing system or remove procedural macros from Rust.

We could have an AST-based (rather than token-based) system. This has major
backwards compatibility issues.

We could allow pluging in at later stages of compilation, giving macros access
to type information, etc. This would allow some really interesting tools.
However, it has some large downsides - it complicates the whole compilation
process (not just the macro system), it pollutes the whole compiler with macro
knowledge, rather than containing it in the frontend, it complicates the design
of the interface between the compiler and macro, and (I believe) the use cases
are better addressed by compiler plug-ins or tools based on the compiler (the
latter can be written today, the former require more work on an interface to the
compiler to be practical).

We could use the `macro` keyword rather than the `fn` keyword to declare a
macro. We would then not require a `#[proc_macro]` attribute.

We could use `#[macro]` instead of `#[proc_macro]` (and similarly for the other
attributes). This would require making `macro` a contextual keyword.

We could have a dedicated syntax for procedural macros, similar to the
`macro_rules` syntax for macros by example. Since a procedural macro is really
just a Rust function, I believe using a function is better. I have also not been
able to come up with (or seen suggestions for) a good alternative syntax. It
seems reasonable to expect to write Rust macros in Rust (although there is
nothing stopping a macro author from using FFI and some other language to write
part or all of a macro).

For attribute-like macros on items, it would be nice if we could skip parsing
the annotated item until after macro expansion. That would allow for more
flexible macros, since the input would not be constrained to Rust syntax. However,
this would require identifying items from tokens, rather than from the AST, which
would require additional rules on token trees and may not be possible.


# Unresolved questions
[unresolved]: #unresolved-questions

### Linking model

Currently, procedural macros are dynamically linked with the compiler. This
prevents the compiler being statically linked, which is sometimes desirable. An
alternative architecture would have procedural macros compiled as independent
programs and have them communicate with the compiler via IPC.

This would have the advantage of allowing static linking for the compiler and
would prevent procedural macros from crashing the main compiler process.
However, designing a good IPC interface is complicated because there is a lot of
data that might be exchanged between the compiler and the macro.

I think we could first design the syntax, interfaces, etc. and later evolve into
a process-separated model (if desired). However, if this is considered an
essential feature of macro reform, then we might want to consider the interfaces
more thoroughly with this in mind.

A step in this direction might be to run the macro in its own thread, but in the
compiler's process.

### Interactions with constant evaluation

Both procedural macros and constant evaluation are mechanisms for running Rust
code at compile time. Currently, and under the proposed design, they are
considered completely separate features. There might be some benefit in letting
them interact.


### Inline procedural macros

It would nice to allow procedural macros to be defined in the crate in which
they are used, as well as in separate crates (mentioned above). This complicates
things since it breaks the invariant that a crate is designed to be used at
either compile-time or runtime. I leave it for the future.


### Specification of the macro definition function signatures

As proposed, the signatures of functions used as macro definitions are hard-
wired into the compiler. It would be more flexible to allow them to be specified
by a lang-item. I'm not sure how beneficial this would be, since a change to the
signature would require changing much of the procedural macro system. I propose
leaving them hard-wired, unless there is a good use case for the more flexible
approach.


### Specifying delimiters

Under this RFC, a function-like macro use may use either parentheses, braces, or
square brackets. The choice of delimiter does not affect the semantics of the
macro (the rules requiring braces or a semi-colon for macro uses in item position
still apply).

Which delimiter was used should be available to the macro implementation via the
`MacroContext`. I believe this is maximally flexible - the macro implementation
can throw an error if it doesn't like the delimiters used.

We might want to allow the compiler to restrict the delimiters. Alternatively,
we might want to hide the information about the delimiter from the macro author,
so as not to allow errors regarding delimiter choice to affect the user.
