- Feature Name: libsyntax2
- Start Date: 2017-12-30
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)


>I think the lack of reusability comes in object-oriented languages,
>not functional languages. Because the problem with object-oriented
>languages is they’ve got all this implicit environment that they
>carry around with them. You wanted a banana but what you got was a
>gorilla holding the banana and the entire jungle.
>
>If you have referentially transparent code, if you have pure
>functions — all the data comes in its input arguments and everything
>goes out and leave no state behind — it’s incredibly reusable.
>
> **Joe Armstrong**

# Summary
[summary]: #summary

The long-term plan is to rewrite libsyntax parser and syntax tree data
structure to create a software component independent of the rest of
rustc compiler and suitable for the needs of IDEs and code
editors. This RFCs is the first step of this plan, whose goal is to
find out if this is possible at least in theory. If it is possible,
the next steps would be a prototype implementation as a crates.io
crate and a separate RFC for integrating the prototype with rustc,
other tools, and eventual libsyntax removal.

Note that this RFC does not propose to stabilize any API for working
with rust syntax: the semver version of the hypothetical library would
be `0.1.0`.


# Motivation
[motivation]: #motivation

## Reusability

In theory, parsing can be a pure function, which takes a `&str` as an
input, and produces a `ParseTree` as an output.

This is great for reusability: for example, you can compile this
function to WASM and use it for fast client-side validation of syntax
on the rust playground, or you can develop tools like `rustfmt` on
stable Rust outside of rustc repository, or you can embed the parser
into your favorite IDE or code editor.

This is also great for correctness: with such simple interface, it's
possible to write property-based tests to thoroughly compare two
different implementations of the parser. It's also straightforward to
create a comprehensive test suite, because all the inputs and outputs
are trivially serializable to human-readable text.

Another benefit is performance: with this signature, you can cache a
parse tree for each file, with trivial strategy for cache invalidation
(invalidate an entry when the underling file changes). On top of such
a cache it is possible to build a smart code indexer which maintains
the set of symbols in the project, watches files for changes and
automatically reindexes only changed files.

Unfortunately, the current libsyntax is far from this ideal. For
example, even the lexer makes use of the `FileMap` which is
essentially a global state of the compiler which represents all know
files. As a data point, it turned out to be easier to move `rustfmt`
inside of main `rustc` repository than to move libsyntax outside!


## IDE support

There is one big difference in how IDEs and compilers typically treat
source code. 

In the compiler, it is convenient to transform the source
code into Abstract Syntax Tree form, which is independent of the
surface syntax. For example, it's convenient to discard comments,
whitespaces and desugar some syntactic constructs in terms of the
simpler ones.

In contrast, for IDEs it is crucial to have a lossless view of the
source code because, for example, it's important to preserve comments
during refactorings. Ideally, IDEs should be able to incrementally
relex and reparse the file as the user types, because syntax tree is
necessary to correctly handle certain code-editing actions like
autoindentation or joining lines. IDE also must be able to produce
partial parse trees when some input is missing or invalid.

Currently rustc uses the AST approach, which preserves the source code
information to some extent by storing spans in the AST.



# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Not applicable.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This section proposes a new syntax tree data structure, which should
be suitable for both compiler and IDE. It is heavily inspired by [PSI]
data structure which used in [IntelliJ] based IDEs and in the [Kotlin]
compiler.


[PSI]: http://www.jetbrains.org/intellij/sdk/docs/reference_guide/custom_language_support/implementing_parser_and_psi.html
[IntelliJ]: https://github.com/JetBrains/intellij-community/
[Kotlin]: https://kotlinlang.org/


The main idea is to store the minimal amount of information in the
tree itself, and instead lean heavily on the source code string for
the actual data about identifier names, constant values etc.

All nodes in the tree are of the same type and store a constant for
the syntactic category of the element and a range in the source code.

Here is a minimal implementation of this data structure with some Rust
syntactic categories


```rust
```

Here is a rust snippet and the corresponding parse tree:

```rust
struct Foo {
    field1: u32,
    &
    // non-doc comment
    field2:
}
```


```
FILE
  STRUCT_DEF
    STRUCT_KW
	WHITESPACE
	IDENT
	WHITESPACE
	L_CURLY
	WHITESPACE
	FIELD_DEF
	  IDENT
	  COLON
	  WHITESPACE
	  TYPE_REF
	    IDENT
	COMMA
	WHITESPACE
	ERROR
	  AMP
	WHITESPACE
	FIELD_DEF
	  LINE_COMMENT
	  WHITESPACE
	  IDENT
	  COLON
	  ERROR
	WHITESPACE
	R_CURLY
```

Note several features of the tree:

* All whitespace and comments are explicitly accounted for.

* The node for `STRUCT_DEF` contains the error element for `&`, but
  still represents the following field correctly.
  
* The second field of the struct is incomplete: `FIELD_DEF` node for
  it contains an `ERROR` element, but nevertheless has the correct
  `NodeKind`.
  
* The non-documenting comment is correctly attached to the following
  field.
  


# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and alternatives
[alternatives]: #alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Unresolved questions
[unresolved]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
