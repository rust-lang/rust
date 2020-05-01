# Macro expansion

> `librustc_ast`, `librustc_expand`, and `librustc_builtin_macros` are all undergoing
> refactoring, so some of the links in this chapter may be broken.

Rust has a very powerful macro system. In the previous chapter, we saw how the
parser sets aside macros to be expanded. This chapter is about the process of
expanding those macros iteratively until we have a complete AST for our crate
with no unexpanded macros (or a compile error).

First, we will discuss the algorithm that expands and integrates macro output
into ASTs. Next, we will take a look at how hygiene data is collected. Finally,
we will look at the specifics of expanding different types of macros.

## Expansion and AST Integration

First of all, expansion happens at the crate level. Given a raw source code for
a crate, the compiler will produce a massive AST with all macros expanded, all
modules inlined, etc.

The primary entry point for this process is the
[`MacroExpander::fully_expand_fragment`][fef] method. Usually, we run this
method on a whole crate. If it is not run on a full crate, it means we are
doing _eager macro expansion_. Eager expansion means that we expand the
arguments of a macro invocation before the macro invocation itself. This is
implemented only for a few special built-in macros that expect literals (it's
not a generally available feature of Rust). Eager expansion generally performs
a subset of the things that lazy (normal) expansion does, so we will focus on
lazy expansion for the rest of this chapter.

As an example, consider the following:

```rust,ignore
macro bar($i: ident) { $i }
macro foo($i: ident) { $i }

foo!(bar!(baz));
```

A lazy expansion would expand `foo!` first. An eager expansion would expand
`bar!` first. Implementing eager expansion more generally would be challenging,
but we implement it for a few special built-in macros for the sake of user
experience.

At a high level, [`fully_expand_fragment`][fef] works in iterations. We keep a
queue of unresolved macro invocations (that is, macros we haven't found the
definition of yet). We repeatedly try to pick a macro from the queue, resolve
it, expand it, and integrate it back. If we can't make progress in an
iteration, this represents a compile error.  Here is the [algorithm][original]:

[fef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/struct.MacroExpander.html#method.fully_expand_fragment
[original]: https://github.com/rust-lang/rust/pull/53778#issuecomment-419224049

0. Initialize an `queue` of unresolved macros.
1. Repeat until `queue` is empty (or we make no progress, which is an error):
    0. [Resolve](./name-resolution.md) imports in our partially built crate as
       much as possible.
    1. Collect as many macro invocations as possible from our partially built
       crate (fn-like, attributes, derives) and add them to the queue.
    2. Dequeue the first element, and attempt to resolve it.
    3. If it's resolved:
        0. Run the macro's expander function that consumes tokens or AST and
           produces tokens or AST (depending on the macro kind).
            - At this point, we know everything about the macro itself and can
              call `set_expn_data` to fill in its properties in the global data
              -- that is the hygiene data associated with `ExpnId`. (See [the
              "Hygiene" section below][hybelow]).
        1. Integrate that piece of AST into the big existing partially built
           AST. This is essentially where the "token-like mass" becomes a
           proper set-in-stone AST with side-tables. It happens as follows:
            - If the macro produces tokens (e.g. a proc macro), we parse into
              an AST, which may produce parse errors.
            - During expansion, we create `SyntaxContext`s (heirarchy 2). (See
              [the "Hygiene" section below][hybelow])
            - These three passes happen one after another on every AST fragment
              freshly expanded from a macro:
                - [`NodeId`]s are assigned by [`InvocationCollector`]. This
                  also collects new macro calls from this new AST piece and
                  adds them to the queue.
                - ["Def paths"][defpath] are created and [`DefId`]s are
                  assigned to them by [`DefCollector`].
                - Names are put into modules (from the resolver's point of
                  view) by [`BuildReducedGraphVisitor`].
        2. After expanding a single macro and integrating its output, continue
           to the next iteration of [`fully_expand_fragment`][fef].
    4. If it's not resolved:
        0. Put the macro back in the queue
        1. Continue to next iteration...

[defpaths]: https://rustc-dev-guide.rust-lang.org/hir.html?highlight=def,path#identifiers-in-the-hir
[`NodeId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html
[`InvocationCollector`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/struct.InvocationCollector.html
[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`DefCollector`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/def_collector/struct.DefCollector.html
[`BuildReducedGraphVisitor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/build_reduced_graph/struct.BuildReducedGraphVisitor.html
[hybelow]: #hygiene-and-heirarchies

If we make no progress in an iteration, then we have reached a compilation
error (e.g. an undefined macro). We attempt to recover from failures
(unresolved macros or imports) for the sake of diagnostics. This allows
compilation to continue past the first error, so that we can report more errors
at a time. Recovery can't cause compilation to suceed. We know that it will
fail at this point. The recovery happens by expanding unresolved macros into
[`ExprKind::Err`][err].

[err]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.ExprKind.html#variant.Err

Notice that name resolution is involved here: we need to resolve imports and
macro names in the above algorithm. However, we don't try to resolve other
names yet. This happens later, as we will see in the [next
chapter](./name-resolution.md).

## Hygiene and Heirarchies

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

[code_dir]: https://github.com/rust-lang/rust/tree/master/src/librustc_expand/mbe
[code_mp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser
[code_mr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_rules
[code_parse_int]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser/fn.parse_tt.html
[parsing]: ./the-parser.html

The context is attached to AST nodes. All AST nodes generated by macros have
context attached. Additionally, there may be other nodes that have context
attached, such as some desugared syntax (non-macro-expanded nodes are
considered to just have the "root" context, as described below).

Because macros invocations and definitions can be nested, the syntax context of
a node must be a heirarchy. For example, if we expand a macro and there is
another macro invocation or definition in the generated output, then the syntax
context should reflex the nesting.

However, it turns out that there are actually a few types of context we may
want to track for different purposes. Thus, there not just one but _three_
expansion heirarchies that together comprise the hygiene information for a
crate.

All of these heirarchies need some sort of "macro ID" to identify individual
elements in the chain of expansions. This ID is [`ExpnId`].  All macros receive
an integer ID, assigned continuously starting from 0 as we discover new macro
calls.  All heirarchies start at [`ExpnId::root()`][rootid], which is its own
parent.

The actual heirarchies are stored in [`HygieneData`][hd], and all of the
hygiene-related algorithms are implemented in [`rustc_span::hygiene`][hy], with
the exception of some hacks [`Resolver::resolve_crate_root`][hacks].

[`ExpnId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html
[rootid]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html#method.root
[hd]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.HygieneData.html
[hy]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/index.html
[hacks]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html#method.resolve_crate_root

### The Expansion Order Heirarchy

The first heirarchy tracks the order of expansions, i.e., when a macro
invocation is in the output of another macro.

Here, the children in the heirarchy will be the "innermost" tokens.
[`ExpnData::parent`][edp] tracks the child -> parent link in this heirarchy.

[edp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnData.html#structfield.parent

For example,

```rust,ignore
macro_rules! foo { () => { println!(); } }

fn main() { foo!(); }
```

In this code, the AST nodes that are finally generated would have heirarchy:

```
root
    expn_id_foo
        expn_id_println
```

### The Macro Definition Heirarchy

The second heirarchy tracks the order of macro definitions, i.e., when we are
expanding one macro another macro definition is revealed in its output.  This
one is a bit tricky and more complex than the other two heirarchies.

Here, [`SyntaxContextData::parent`][scdp] is the child -> parent link here.
[`SyntaxContext`][sc] is the whole chain in this hierarchy, and
[`SyntaxContextData::outer_expns`][scdoe] are individual elements in the chain.
The "chaining operator" is [`SyntaxContext::apply_mark`][am] in compiler code.

[scdp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContextData.html#structfield.parent
[sc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html
[scdoe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContextData.html#structfield.outer_expn
[am]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html#method.apply_mark

For built-in macros, we use the context:
`SyntaxContext::empty().apply_mark(expn_id)`, and such macros are considered to
be defined at the heirarchy root. We do the same for proc-macros because we
haven't implemented cross-crate hygiene yet.

If the token had context `X` before being produced by a macro then after being
produced by the macro it has context `X -> macro_id`. Here are some examples:

Example 0:

```rust,ignore
macro m() { ident }

m!();
```

Here `ident` originally has context [`SyntaxContext::root()`][scr]. `ident` has
context `ROOT -> id(m)` after it's produced by `m`.

[scr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html#method.root


Example 1:

```rust,ignore
macro m() { macro n() { ident } }

m!();
n!();
```
In this example the `ident` has context `ROOT` originally, then `ROOT -> id(m)`
after the first expansion, then `ROOT -> id(m) -> id(n)`.

Example 2:

Note that these chains are not entirely determined by their last element, in
other words `ExpnId` is not isomorphic to `SyntaxContext`.

```rust,ignore
macro m($i: ident) { macro n() { ($i, bar) } }

m!(foo);
```

After all expansions, `foo` has context `ROOT -> id(n)` and `bar` has context
`ROOT -> id(m) -> id(n)`.

Finally, one last thing to mention is that currently, this heirarchy is subject
to the ["context transplantation hack"][hack]. Basically, the more modern (and
experimental) `macro` macros have stronger hygiene than the older MBE system,
but this can result in weird interactions between the two. The hack is intended
to make things "just work" for now.

[hack]: https://github.com/rust-lang/rust/pull/51762#issuecomment-401400732

### The Call-site Heirarchy

The third and final heirarchy tracks the location of macro invocations.

In this heirarchy [`ExpnData::call_site`][callsite] is the child -> parent link.

[callsite]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnData.html#structfield.call_site

Here is an example:

```rust,ignore
macro bar($i: ident) { $i }
macro foo($i: ident) { $i }

foo!(bar!(baz));
```

For the `baz` AST node in the final output, the first heirarchy is `ROOT ->
id(foo) -> id(bar) -> baz`, while the third heirarchy is `ROOT -> baz`.

## Producing Macro Output

Above, we saw how the output of a macro is integrated into the AST for a crate,
and we also saw how th e hygiene data for a crate is generated. But how do we
actually produce the output of a macro? It depends on the type of macro.

There are two types of macros in Rust:
`macro_rules!` macros (a.k.a. "Macros By Example" (MBE)) and procedural macros
(or "proc macros"; including custom derives). During the parsing phase, the normal
Rust parser will set aside the contents of macros and their invocations. Later,
macros are expanded using these portions of the code.

## Macros By Example

MBEs have their own parser distinct from the normal Rust parser. When macros
are expanded, we may invoke the MBE parser to parse and expand a macro.  The
MBE parser, in turn, may call the normal Rust parser when it needs to bind a
metavariable (e.g.  `$my_expr`) while parsing the contents of a macro
invocation. The code for macro expansion is in
[`src/librustc_expand/mbe/`][code_dir].

### Example

It's helpful to have an example to refer to. For the remainder of this chapter,
whenever we refer to the "example _definition_", we mean the following:

```rust,ignore
macro_rules! printer {
    (print $mvar:ident) => {
        println!("{}", $mvar);
    };
    (print twice $mvar:ident) => {
        println!("{}", $mvar);
        println!("{}", $mvar);
    };
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

### The MBE parser

There are two parts to MBE expansion: parsing the definition and parsing the
invocations. Interestingly, both are done by the macro parser.

Basically, the MBE parser is like an NFA-based regex parser. It uses an
algorithm similar in spirit to the [Earley parsing
algorithm](https://en.wikipedia.org/wiki/Earley_parser). The macro parser is
defined in [`src/librustc_expand/mbe/macro_parser.rs`][code_mp].

The interface of the macro parser is as follows (this is slightly simplified):

```rust,ignore
fn parse_tt(
    parser: &mut Cow<Parser>,
    ms: &[TokenTree],
) -> NamedParseResult
```

We use these items in macro parser:

- `parser` is a reference to the state of a normal Rust parser, including the
  token stream and parsing session. The token stream is what we are about to
  ask the MBE parser to parse. We will consume the raw stream of tokens and
  output a binding of metavariables to corresponding token trees. The parsing
  session can be used to report parser errros.
- `ms` a _matcher_. This is a sequence of token trees that we want to match
  the token stream against.

In the analogy of a regex parser, the token stream is the input and we are matching it
against the pattern `ms`. Using our examples, the token stream could be the stream of
tokens containing the inside of the example invocation `print foo`, while `ms`
might be the sequence of token (trees) `print $mvar:ident`.

The output of the parser is a `NamedParseResult`, which indicates which of
three cases has occurred:

- Success: the token stream matches the given matcher `ms`, and we have produced a binding
  from metavariables to the corresponding token trees.
- Failure: the token stream does not match `ms`. This results in an error message such as
  "No rule expected token _blah_".
- Error: some fatal error has occurred _in the parser_. For example, this
  happens if there are more than one pattern match, since that indicates
  the macro is ambiguous.

The full interface is defined [here][code_parse_int].

The macro parser does pretty much exactly the same as a normal regex parser with
one exception: in order to parse different types of metavariables, such as
`ident`, `block`, `expr`, etc., the macro parser must sometimes call back to the
normal Rust parser.

As mentioned above, both definitions and invocations of macros are parsed using
the macro parser. This is extremely non-intuitive and self-referential. The code
to parse macro _definitions_ is in
[`src/librustc_expand/mbe/macro_rules.rs`][code_mr]. It defines the pattern for
matching for a macro definition as `$( $lhs:tt => $rhs:tt );+`. In other words,
a `macro_rules` definition should have in its body at least one occurrence of a
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
in [`src/librustc_expand/mbe/macro_parser.rs`][code_mp].

### `macro`s and Macros 2.0

There is an old and mostly undocumented effort to improve the MBE system, give
it more hygiene-related features, better scoping and visibility rules, etc. There
hasn't been a lot of work on this recently, unfortunately. Internally, `macro`
macros use the same machinery as today's MBEs; they just have additional
syntactic sugar and are allowed to be in namespaces.

## Procedural Macros

Precedural macros are also expanded during parsing, as mentioned above.
However, they use a rather different mechanism. Rather than having a parser in
the compiler, procedural macros are implemented as custom, third-party crates.
The compiler will compile the proc macro crate and specially annotated
functions in them (i.e. the proc macro itself), passing them a stream of tokens.

The proc macro can then transform the token stream and output a new token
stream, which is synthesized into the AST.

It's worth noting that the token stream type used by proc macros is _stable_,
so `rustc` does not use it internally (since our internal data structures are
unstable).

TODO: more here.

### Custom Derive

Custom derives are a special type of proc macro.

TODO: more?

## Notes from petrochenkov discussion

TODO: sprinkle these links around the chapter...

Where to find the code:
- librustc_span/hygiene.rs - structures related to hygiene and expansion that are kept in global data (can be accessed from any Ident without any context)
- librustc_span/lib.rs - some secondary methods like macro backtrace using primary methods from hygiene.rs
- librustc_builtin_macros - implementations of built-in macros (including macro attributes and derives) and some other early code generation facilities like injection of standard library imports or generation of test harness.
- librustc_ast/config.rs - implementation of cfg/cfg_attr (they treated specially from other macros), should probably be moved into librustc_ast/ext.
- librustc_ast/tokenstream.rs + librustc_ast/parse/token.rs - structures for compiler-side tokens, token trees, and token streams.
- librustc_ast/ext - various expansion-related stuff
- librustc_ast/ext/base.rs - basic structures used by expansion
- librustc_ast/ext/expand.rs - some expansion structures and the bulk of expansion infrastructure code - collecting macro invocations, calling into resolve for them, calling their expanding functions, and integrating the results back into AST
- librustc_ast/ext/placeholder.rs - the part of expand.rs responsible for "integrating the results back into AST" basicallly, "placeholder" is a temporary AST node replaced with macro expansion result nodes
- librustc_ast/ext/builer.rs - helper functions for building AST for built-in macros in librustc_builtin_macros (and user-defined syntactic plugins previously), can probably be moved into librustc_builtin_macros these days
- librustc_ast/ext/proc_macro.rs + librustc_ast/ext/proc_macro_server.rs - interfaces between the compiler and the stable proc_macro library, converting tokens and token streams between the two representations and sending them through C ABI
- librustc_ast/ext/tt - implementation of macro_rules, turns macro_rules DSL into something with signature Fn(TokenStream) -> TokenStream that can eat and produce tokens, @mark-i-m knows more about this
- librustc_resolve/macros.rs - resolving macro paths, validating those resolutions, reporting various "not found"/"found, but it's unstable"/"expected x, found y" errors
- librustc_middle/hir/map/def_collector.rs + librustc_resolve/build_reduced_graph.rs - integrate an AST fragment freshly expanded from a macro into various parent/child structures like module hierarchy or "definition paths"

Primary structures:
- HygieneData - global piece of data containing hygiene and expansion info that can be accessed from any Ident without any context
- ExpnId - ID of a macro call or desugaring (and also expansion of that call/desugaring, depending on context)
- ExpnInfo/InternalExpnData - a subset of properties from both macro definition and macro call available through global data
- SyntaxContext - ID of a chain of nested macro definitions (identified by ExpnIds)
- SyntaxContextData - data associated with the given SyntaxContext, mostly a cache for results of filtering that chain in different ways
- Span - a code location + SyntaxContext
- Ident - interned string (Symbol) + Span, i.e. a string with attached hygiene data
- TokenStream - a collection of TokenTrees
- TokenTree - a token (punctuation, identifier, or literal) or a delimited group (anything inside ()/[]/{})
- SyntaxExtension - a lowered macro representation, contains its expander function transforming a tokenstream or AST into tokenstream or AST + some additional data like stability, or a list of unstable features allowed inside the macro.
- SyntaxExtensionKind - expander functions may have several different signatures (take one token stream, or two, or a piece of AST, etc), this is an enum that lists them
- ProcMacro/TTMacroExpander/AttrProcMacro/MultiItemModifier - traits representing the expander signatures (TODO: change and rename the signatures into something more consistent)
- Resolver - a trait used to break crate dependencies (so resolver services can be used in librustc_ast, despite librustc_resolve and pretty much everything else depending on librustc_ast)
- ExtCtxt/ExpansionData - various intermediate data kept and used by expansion infra in the process of its work
- AstFragment - a piece of AST that can be produced by a macro (may include multiple homogeneous AST nodes, like e.g. a list of items)
- Annotatable - a piece of AST that can be an attribute target, almost same thing as AstFragment except for types and patterns that can be produced by macros but cannot be annotated with attributes (TODO: Merge into AstFragment)
- MacResult - a "polymorphic" AST fragment, something that can turn into a different AstFragment depending on its context (aka AstFragmentKind - item, or expression, or pattern etc.)
- Invocation/InvocationKind - a structure describing a macro call, these structures are collected by the expansion infra (InvocationCollector), queued, resolved, expanded when resolved, etc.
