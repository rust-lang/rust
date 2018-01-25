# Macro expansion

Macro expansion happens during parsing. `rustc` has two parsers, in fact: the
normal Rust parser, and the macro parser. During the parsing phase, the normal
Rust parser will call into the macro parser when it encounters a macro. The
macro parser, in turn, may call back out to the Rust parser when it needs to
bind a metavariable (e.g. `$my_expr`). There are a few aspects of this system to
be explained. The code for macro expansion is in `src/libsyntax/ext/tt/`.

### The macro parser

Basically, the macro parser is like an NFA-based regex parser. It uses an
algorithm similar in spirit to the [Earley parsing
algorithm](https://en.wikipedia.org/wiki/Earley_parser). The macro parser is
defined in `src/libsyntax/ext/tt/macro_parser.rs`.

In a traditional NFA-based parser, one common approach is to have some pattern
which we are trying to match an input against. Moreover, we may try to capture
some portion of the input and bind it to variable in the pattern. For example:
suppose we have a pattern (borrowing Rust macro syntax) such as `a $b:ident a`
-- that is, an `a` token followed by an `ident` token followed by another `a`
token. Given an input `a foo a`, the _metavariable_ `$b` would bind to the
`ident` `foo`. On the other hand, an input `a foo b` would be rejected as a
parse failure because the pattern `a <ident> a` cannot match `a foo b` (or as
the compiler would put it, "no rules expected token `b`").

The macro parser does pretty much exactly that with one exception: in order to
parse different types of metavariables, such as `ident`, `block`, `expr`, etc.,
the macro parser must sometimes call back to the normal Rust parser.

Interestingly, both definitions and invokations of macros are parsed using the
macro parser. This is extremely non-intuitive and self-referential. The code to
parse macro _definitions_ is in `src/libsyntax/ext/tt/macro_rules.rs`. It
defines the pattern for matching for a macro definition as `$( $lhs:tt =>
$rhs:tt );+`. In other words, a `macro_rules` defintion should have in its body
at least one occurence of a token tree followed by `=>` followed by another
token tree. When the compiler comes to a `macro_rules` definition, it uses this
pattern to match the two token trees per rule in the definition of the macro
_using the macro parser itself_.

When the compiler comes to a macro invokation, it needs to parse that
invokation. This is also known as _macro expansion_. The same NFA-based macro
parser is used that is described above. Notably, the "pattern" (or _matcher_)
used is the first token tree extracted from the rules of the macro _definition_.
In other words, given some pattern described by the _definition_ of the macro,
we want to match the contents of the _invokation_ of the macro.

The algorithm is exactly the same, but when the macro parser comes to a place in
the current matcher where it needs to match a _non-terminal_ (i.e. a
metavariable), it calls back to the normal Rust parser to get the contents of
that non-terminal. Then, the macro parser proceeds in parsing as normal.

For more information about the macro parser's implementation, see the comments
in `src/libsyntax/ext/tt/macro_parser.rs`.

### Hygiene

TODO

### Procedural Macros

TODO

### Custom Derive

TODO
