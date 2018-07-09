# Macro expansion

Macro expansion happens during parsing. `rustc` has two parsers, in fact: the
normal Rust parser, and the macro parser. During the parsing phase, the normal
Rust parser will set aside the contents of macros and their invocations. Later,
before name resolution, macros are expanded using these portions of the code.
The macro parser, in turn, may call the normal Rust parser when it needs to
bind a metavariable (e.g.  `$my_expr`) while parsing the contents of a macro
invocation. The code for macro expansion is in
[`src/libsyntax/ext/tt/`][code_dir]. This chapter aims to explain how macro
expansion works.

### Example

It's helpful to have an example to refer to. For the remainder of this chapter,
whenever we refer to the "example _definition_", we mean the following:

```rust,ignore
macro_rules! printer {
    (print $mvar:ident) => {
        println!("{}", $mvar);
    }
    (print twice $mvar:ident) => {
        println!("{}", $mvar);
        println!("{}", $mvar);
    }
}
```

`$mvar` is called a _metavariable_. Unlike normal variables, rather than
binding to a value in a computation, a metavariable binds _at compile time_ to
a tree of _tokens_.  A _token_ is a single "unit" of the grammar, such as an
identifier (e.g. `foo`) or punctuation (e.g. `=>`). There are also other
special tokens, such as `EOF`, which indicates that there are no more tokens.
Token trees resulting from paired parentheses-like characters (`(`...`)`,
`[`...`]`, and `{`...`}`) – they include the open and close and all the tokens
in between (we do require that parentheses-like characters be balanced). Having
macro expansion operate on token streams rather than the raw bytes of a source
file abstracts away a lot of complexity. The macro expander (and much of the
rest of the compiler) doesn't really care that much about the exact line and
column of some syntactic construct in the code; it cares about what constructs
are used in the code. Using tokens allows us to care about _what_ without
worrying about _where_. For more information about tokens, see the
[Parsing][parsing] chapter of this book.

Whenever we refer to the "example _invocation_", we mean the following snippet:

```rust,ignore
printer!(print foo); // Assume `foo` is a variable defined somewhere else...
```

The process of expanding the macro invocation into the syntax tree
`println!("{}", foo)` and then expanding that into a call to `Display::fmt` is
called _macro expansion_, and it is the topic of this chapter.

### The macro parser

There are two parts to macro expansion: parsing the definition and parsing the
invocations. Interestingly, both are done by the macro parser.

Basically, the macro parser is like an NFA-based regex parser. It uses an
algorithm similar in spirit to the [Earley parsing
algorithm](https://en.wikipedia.org/wiki/Earley_parser). The macro parser is
defined in [`src/libsyntax/ext/tt/macro_parser.rs`][code_mp].

The interface of the macro parser is as follows (this is slightly simplified):

```rust,ignore
fn parse(
    sess: ParserSession,
    tts: TokenStream,
    ms: &[TokenTree]
) -> NamedParseResult
```

In this interface:

- `sess` is a "parsing session", which keeps track of some metadata. Most
  notably, this is used to keep track of errors that are generated so they can
  be reported to the user.
- `tts` is a stream of tokens. The macro parser's job is to consume the raw
  stream of tokens and output a binding of metavariables to corresponding token
  trees.
- `ms` a _matcher_. This is a sequence of token trees that we want to match
  `tts` against.

In the analogy of a regex parser, `tts` is the input and we are matching it
against the pattern `ms`. Using our examples, `tts` could be the stream of
tokens containing the inside of the example invocation `print foo`, while `ms`
might be the sequence of token (trees) `print $mvar:ident`.

The output of the parser is a `NamedParseResult`, which indicates which of
three cases has occured:

- Success: `tts` matches the given matcher `ms`, and we have produced a binding
  from metavariables to the corresponding token trees.
- Failure: `tts` does not match `ms`. This results in an error message such as
  "No rule expected token _blah_".
- Error: some fatal error has occured _in the parser_. For example, this happens
  if there are more than one pattern match, since that indicates the macro is
  ambiguous.

The full interface is defined [here][code_parse_int].

The macro parser does pretty much exactly the same as a normal regex parser with
one exception: in order to parse different types of metavariables, such as
`ident`, `block`, `expr`, etc., the macro parser must sometimes call back to the
normal Rust parser.

As mentioned above, both definitions and invocations of macros are parsed using
the macro parser. This is extremely non-intuitive and self-referential. The code
to parse macro _definitions_ is in
[`src/libsyntax/ext/tt/macro_rules.rs`][code_mr]. It defines the pattern for
matching for a macro definition as `$( $lhs:tt => $rhs:tt );+`. In other words,
a `macro_rules` defintion should have in its body at least one occurence of a
token tree followed by `=>` followed by another token tree. When the compiler
comes to a `macro_rules` definition, it uses this pattern to match the two token
trees per rule in the definition of the macro _using the macro parser itself_.
In our example definition, the metavariable `$lhs` would match the patterns of
both arms: `(print $mvar:ident)` and `(print twice $mvar:ident)`.  And `$rhs`
would match the bodies of both arms: `{ println!("{}", $mvar); }` and `{
println!("{}", $mvar); println!("{}", $mvar); }`. The parser would keep this
knowledge around for when it needs to expand a macro invocation.

When the compiler comes to a macro invocation, it parses that invocation using
the same NFA-based macro parser that is described above. However, the matcher
used is the first token tree (`$lhs`) extracted from the arms of the macro
_definition_. Using our example, we would try to match the token stream `print
foo` from the invocation against the matchers `print $mvar:ident` and `print
twice $mvar:ident` that we previously extracted from the definition.  The
algorithm is exactly the same, but when the macro parser comes to a place in the
current matcher where it needs to match a _non-terminal_ (e.g. `$mvar:ident`),
it calls back to the normal Rust parser to get the contents of that
non-terminal. In this case, the Rust parser would look for an `ident` token,
which it finds (`foo`) and returns to the macro parser. Then, the macro parser
proceeds in parsing as normal. Also, note that exactly one of the matchers from
the various arms should match the invocation; if there is more than one match,
the parse is ambiguous, while if there are no matches at all, there is a syntax
error.

For more information about the macro parser's implementation, see the comments
in [`src/libsyntax/ext/tt/macro_parser.rs`][code_mp].

### Hygiene

If you have ever used C/C++ preprocessor macros, you know that there are some
annoying and hard-to-debug gotchas! For example, consider the following C code:

```c
#define DEFINE_FOO struct Bar {int x;}; struct Foo {Bar bar;};

// Then, somewhere else
struct Bar {
    ...
};

DEFINE_FOO
```

Most people avoid writing C like this – and for good reason: it doesn't
compile. The `struct Bar` defined by the macro clashes names with the `struct
Bar` defined in the code. Consider also the following example:

```c
#define DO_FOO(x) {\
    int y = 0;\
    foo(x, y);\
    }

// Then elsewhere
int y = 22;
DO_FOO(y);
```

Do you see the problem? We wanted to generate a call `foo(22, 0)`, but instead
we got `foo(0, 0)` because the macro defined its own `y`!

These are both examples of _macro hygiene_ issues. _Hygiene_ relates to how to
handle names defined _within a macro_. In particular, a hygienic macro system
prevents errors due to names introduced within a macro. Rust macros are hygienic
in that they do not allow one to write the sorts of bugs above.

At a high level, hygiene within the rust compiler is accomplished by keeping
track of the context where a name is introduced and used. We can then
disambiguate names based on that context. Future iterations of the macro system
will allow greater control to the macro author to use that context. For example,
a macro author may want to introduce a new name to the context where the macro
was called. Alternately, the macro author may be defining a variable for use
only within the macro (i.e. it should not be visible outside the macro).

In rustc, this "context" is tracked via `Span`s.

TODO: what is call-site hygiene? what is def-site hygiene?

TODO

### Procedural Macros

TODO

### Custom Derive

TODO

TODO: maybe something about macros 2.0?


[code_dir]: https://github.com/rust-lang/rust/tree/master/src/libsyntax/ext/tt
[code_mp]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ext/tt/macro_parser/
[code_mr]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ext/tt/macro_rules/
[code_parse_int]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ext/tt/macro_parser/fn.parse.html
[parsing]: ./the-parser.html
