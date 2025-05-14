# Macro expansion

<!-- toc -->

Rust has a very powerful macro system. In the previous chapter, we saw how
the parser sets aside macros to be expanded (using temporary [placeholders]).
This chapter is about the process of expanding those macros iteratively until
we have a complete [*Abstract Syntax Tree* (AST)][ast] for our crate with no
unexpanded macros (or a compile error).

[ast]: ./ast-validation.md
[placeholders]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/placeholders/index.html

First, we discuss the algorithm that expands and integrates macro output into
ASTs. Next, we take a look at how hygiene data is collected. Finally, we look
at the specifics of expanding different types of macros.

Many of the algorithms and data structures described below are in [`rustc_expand`],
with fundamental data structures in [`rustc_expand::base`][base].

Also of note, `cfg` and `cfg_attr` are treated specially from other macros, and are
handled in [`rustc_expand::config`][cfg].

[`rustc_expand`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/index.html
[base]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/index.html
[cfg]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/config/index.html

## Expansion and AST Integration

Firstly, expansion happens at the crate level. Given a raw source code for
a crate, the compiler will produce a massive AST with all macros expanded, all
modules inlined, etc. The primary entry point for this process is the
[`MacroExpander::fully_expand_fragment`][fef] method. With few exceptions, we
use this method on the whole crate (see ["Eager Expansion"](#eager-expansion)
below for more detailed discussion of edge case expansion issues).

[`rustc_builtin_macros`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_builtin_macros/index.html
[reb]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/build/index.html

At a high level, [`fully_expand_fragment`][fef] works in iterations. We keep a
queue of unresolved macro invocations (i.e. macros we haven't found the
definition of yet). We repeatedly try to pick a macro from the queue, resolve
it, expand it, and integrate it back. If we can't make progress in an
iteration, this represents a compile error.  Here is the [algorithm][original]:

[fef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/struct.MacroExpander.html#method.fully_expand_fragment
[original]: https://github.com/rust-lang/rust/pull/53778#issuecomment-419224049

1. Initialize a `queue` of unresolved macros.
2. Repeat until `queue` is empty (or we make no progress, which is an error):
   1. [Resolve](./name-resolution.md) imports in our partially built crate as
      much as possible.
   2. Collect as many macro [`Invocation`s][inv] as possible from our
      partially built crate (`fn`-like, attributes, derives) and add them to the
      queue.
   3. Dequeue the first element and attempt to resolve it.
   4. If it's resolved:
      1. Run the macro's expander function that consumes a [`TokenStream`] or
         AST and produces a [`TokenStream`] or [`AstFragment`] (depending on
         the macro kind). (A [`TokenStream`] is a collection of [`TokenTree`s][tt],
         each of which are a token (punctuation, identifier, or literal) or a
         delimited group (anything inside `()`/`[]`/`{}`)).
         - At this point, we know everything about the macro itself and can
           call [`set_expn_data`] to fill in its properties in the global
           data; that is the [hygiene] data associated with [`ExpnId`] (see
           [Hygiene][hybelow] below).
      2. Integrate that piece of AST into the currently-existing though
         partially-built AST. This is essentially where the "token-like mass"
         becomes a proper set-in-stone AST with side-tables. It happens as
         follows:
         - If the macro produces tokens (e.g. a proc macro), we parse into
           an AST, which may produce parse errors.
         - During expansion, we create [`SyntaxContext`]s (hierarchy 2) (see
           [Hygiene][hybelow] below).
         - These three passes happen one after another on every AST fragment
           freshly expanded from a macro:
           - [`NodeId`]s are assigned by [`InvocationCollector`]. This
             also collects new macro calls from this new AST piece and
             adds them to the queue.
           - ["Def paths"][defpath] are created and [`DefId`]s are
             assigned to them by [`DefCollector`].
           - Names are put into modules (from the resolver's point of
             view) by [`BuildReducedGraphVisitor`].
      3. After expanding a single macro and integrating its output, continue
         to the next iteration of [`fully_expand_fragment`][fef].
   5. If it's not resolved:
      1. Put the macro back in the queue.
      2. Continue to next iteration...

[`AstFragment`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/enum.AstFragment.html
[`BuildReducedGraphVisitor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/build_reduced_graph/struct.BuildReducedGraphVisitor.html
[`DefCollector`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/def_collector/struct.DefCollector.html
[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`ExpnId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html
[`InvocationCollector`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/struct.InvocationCollector.html
[`NodeId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html
[`set_expn_data`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.LocalExpnId.html#method.set_expn_data
[`SyntaxContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html
[`TokenStream`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/tokenstream/struct.TokenStream.html
[defpath]: hir.md#identifiers-in-the-hir
[hybelow]: #hygiene-and-hierarchies
[hygiene]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/index.html
[inv]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/struct.Invocation.html
[tt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/tokenstream/enum.TokenTree.html

### Error Recovery

If we make no progress in an iteration we have reached a compilation error
(e.g. an undefined macro). We attempt to recover from failures (i.e.
unresolved macros or imports) with the intent of generating diagnostics.
Failure recovery happens by expanding unresolved macros into
[`ExprKind::Err`][err] and allows compilation to continue past the first error
so that `rustc` can report more errors than just the original failure.

[err]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.ExprKind.html#variant.Err

### Name Resolution

Notice that name resolution is involved here: we need to resolve imports and
macro names in the above algorithm. This is done in
[`rustc_resolve::macros`][mresolve], which resolves macro paths, validates
those resolutions, and reports various errors (e.g. "not found", "found, but
it's unstable", "expected x, found y"). However, we don't try to resolve
other names yet. This happens later, as we will see in the chapter: [Name
Resolution](./name-resolution.md).

[mresolve]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/macros/index.html

### Eager Expansion

_Eager expansion_ means we expand the arguments of a macro invocation before
the macro invocation itself. This is implemented only for a few special
built-in macros that expect literals; expanding arguments first for some of
these macro results in a smoother user experience.  As an example, consider
the following:

```rust,ignore
macro bar($i: ident) { $i }
macro foo($i: ident) { $i }

foo!(bar!(baz));
```

A lazy-expansion would expand `foo!` first. An eager-expansion would expand
`bar!` first.

Eager-expansion is not a generally available feature of Rust.  Implementing
eager-expansion more generally would be challenging, so we implement it for a
few special built-in macros for the sake of user-experience.  The built-in
macros are implemented in [`rustc_builtin_macros`], along with some other
early code generation facilities like injection of standard library imports or
generation of test harness. There are some additional helpers for building
AST fragments in [`rustc_expand::build`][reb]. Eager-expansion generally
performs a subset of the things that lazy (normal) expansion does. It is done
by invoking [`fully_expand_fragment`][fef] on only part of a crate (as opposed
to the whole crate, like we normally do).

### Other Data Structures

Here are some other notable data structures involved in expansion and
integration:
- [`ResolverExpand`] - a `trait` used to break crate dependencies. This allows the
  resolver services to be used in [`rustc_ast`], despite [`rustc_resolve`] and
  pretty much everything else depending on [`rustc_ast`].
- [`ExtCtxt`]/[`ExpansionData`] - holds various intermediate expansion
  infrastructure data.
- [`Annotatable`] - a piece of AST that can be an attribute target, almost the same
  thing as [`AstFragment`] except for types and patterns that can be produced by
  macros but cannot be annotated with attributes.
- [`MacResult`] - a "polymorphic" AST fragment, something that can turn into
  a different [`AstFragment`] depending on its [`AstFragmentKind`] (i.e. an item,
  expression, pattern, etc).

[`AstFragment`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/enum.AstFragment.html
[`rustc_ast`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/index.html
[`rustc_resolve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html
[`ResolverExpand`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.ResolverExpand.html
[`ExtCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/struct.ExtCtxt.html
[`ExpansionData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/struct.ExpansionData.html
[`Annotatable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/enum.Annotatable.html
[`MacResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.MacResult.html
[`AstFragmentKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/expand/enum.AstFragmentKind.html

## Hygiene and Hierarchies

If you have ever used the C/C++ preprocessor macros, you know that there are some
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

At a high level, hygiene within the Rust compiler is accomplished by keeping
track of the context where a name is introduced and used. We can then
disambiguate names based on that context. Future iterations of the macro system
will allow greater control to the macro author to use that context. For example,
a macro author may want to introduce a new name to the context where the macro
was called. Alternately, the macro author may be defining a variable for use
only within the macro (i.e. it should not be visible outside the macro).

[code_dir]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_expand/src/mbe
[code_mp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser
[code_mr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_rules
[code_parse_int]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser/struct.TtParser.html#method.parse_tt
[parsing]: ./the-parser.html

The context is attached to AST nodes. All AST nodes generated by macros have
context attached. Additionally, there may be other nodes that have context
attached, such as some desugared syntax (non-macro-expanded nodes are
considered to just have the "root" context, as described below).
Throughout the compiler, we use [`rustc_span::Span`s][span] to refer to code locations.
This struct also has hygiene information attached to it, as we will see later.

[span]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html

Because macros invocations and definitions can be nested, the syntax context of
a node must be a hierarchy. For example, if we expand a macro and there is
another macro invocation or definition in the generated output, then the syntax
context should reflect the nesting.

However, it turns out that there are actually a few types of context we may
want to track for different purposes. Thus, there are not just one but _three_
expansion hierarchies that together comprise the hygiene information for a
crate.

All of these hierarchies need some sort of "macro ID" to identify individual
elements in the chain of expansions. This ID is [`ExpnId`].  All macros receive
an integer ID, assigned continuously starting from 0 as we discover new macro
calls.  All hierarchies start at [`ExpnId::root`][rootid], which is its own
parent.

The [`rustc_span::hygiene`][hy] crate contains all of the hygiene-related algorithms
(with the exception of some hacks in [`Resolver::resolve_crate_root`][hacks])
and structures related to hygiene and expansion that are kept in global data.

The actual hierarchies are stored in [`HygieneData`][hd]. This is a global
piece of data containing hygiene and expansion info that can be accessed from
any [`Ident`] without any context.


[`ExpnId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html
[rootid]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html#method.root
[hd]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.HygieneData.html
[hy]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/index.html
[hacks]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html#method.resolve_crate_root
[`Ident`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Ident.html

### The Expansion Order Hierarchy

The first hierarchy tracks the order of expansions, i.e., when a macro
invocation is in the output of another macro.

Here, the children in the hierarchy will be the "innermost" tokens.  The
[`ExpnData`] struct itself contains a subset of properties from both macro
definition and macro call available through global data.
[`ExpnData::parent`][edp] tracks the child-to-parent link in this hierarchy.

[`ExpnData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnData.html
[edp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnData.html#structfield.parent

For example:

```rust,ignore
macro_rules! foo { () => { println!(); } }

fn main() { foo!(); }
```

In this code, the AST nodes that are finally generated would have hierarchy
`root -> id(foo) -> id(println)`.

### The Macro Definition Hierarchy

The second hierarchy tracks the order of macro definitions, i.e., when we are
expanding one macro another macro definition is revealed in its output.  This
one is a bit tricky and more complex than the other two hierarchies.

[`SyntaxContext`][sc] represents a whole chain in this hierarchy via an ID.
[`SyntaxContextData`][scd] contains data associated with the given
[`SyntaxContext`][sc]; mostly it is a cache for results of filtering that chain in
different ways.  [`SyntaxContextData::parent`][scdp] is the child-to-parent
link here, and [`SyntaxContextData::outer_expns`][scdoe] are individual
elements in the chain.  The "chaining-operator" is
[`SyntaxContext::apply_mark`][am] in compiler code.

A [`Span`][span], mentioned above, is actually just a compact representation of
a code location and [`SyntaxContext`][sc]. Likewise, an [`Ident`] is just an interned
[`Symbol`] + `Span` (i.e. an interned string + hygiene data).

[`Symbol`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html
[scd]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContextData.html
[scdp]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContextData.html#structfield.parent
[sc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html
[scdoe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContextData.html#structfield.outer_expn
[am]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html#method.apply_mark

For built-in macros, we use the context:
[`SyntaxContext::empty().apply_mark(expn_id)`], and such macros are
considered to be defined at the hierarchy root. We do the same for `proc
macro`s because we haven't implemented cross-crate hygiene yet.

[`SyntaxContext::empty().apply_mark(expn_id)`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html#method.apply_mark

If the token had context `X` before being produced by a macro then after being
produced by the macro it has context `X -> macro_id`. Here are some examples:

Example 0:

```rust,ignore
macro m() { ident }

m!();
```

Here `ident` which initially has context [`SyntaxContext::root`][scr] has
context `ROOT -> id(m)` after it's produced by `m`.

[scr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html#method.root

Example 1:

```rust,ignore
macro m() { macro n() { ident } }

m!();
n!();
```

In this example the `ident` has context `ROOT` initially, then `ROOT -> id(m)`
after the first expansion, then `ROOT -> id(m) -> id(n)`.

Example 2:

Note that these chains are not entirely determined by their last element, in
other words [`ExpnId`] is not isomorphic to [`SyntaxContext`][sc].

```rust,ignore
macro m($i: ident) { macro n() { ($i, bar) } }

m!(foo);
```

After all expansions, `foo` has context `ROOT -> id(n)` and `bar` has context
`ROOT -> id(m) -> id(n)`.

Currently this hierarchy for tracking macro definitions is subject to the
so-called ["context transplantation hack"][hack]. Modern (i.e. experimental)
macros have stronger hygiene than the legacy "Macros By Example" (MBE)
system which can result in weird interactions between the two. The hack is
intended to make things "just work" for now.

[`ExpnId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnId.html
[hack]: https://github.com/rust-lang/rust/pull/51762#issuecomment-401400732

### The Call-site Hierarchy

The third and final hierarchy tracks the location of macro invocations.

In this hierarchy [`ExpnData::call_site`][callsite] is the `child -> parent`
link.

[callsite]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/hygiene/struct.ExpnData.html#structfield.call_site

Here is an example:

```rust,ignore
macro bar($i: ident) { $i }
macro foo($i: ident) { $i }

foo!(bar!(baz));
```

For the `baz` AST node in the final output, the expansion-order hierarchy is
`ROOT -> id(foo) -> id(bar) -> baz`, while the call-site hierarchy is `ROOT ->
baz`.

### Macro Backtraces

Macro backtraces are implemented in [`rustc_span`] using the hygiene machinery
in [`rustc_span::hygiene`][hy].

[`rustc_span`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/index.html

## Producing Macro Output

Above, we saw how the output of a macro is integrated into the AST for a crate,
and we also saw how the hygiene data for a crate is generated. But how do we
actually produce the output of a macro? It depends on the type of macro.

There are two types of macros in Rust: 
  1. `macro_rules!` macros (a.k.a. "Macros By Example" (MBE)), and,
  2. procedural macros (proc macros); including custom derives. 
  
During the parsing phase, the normal Rust parser will set aside the contents of
macros and their invocations. Later, macros are expanded using these
portions of the code.

Some important data structures/interfaces here:
- [`SyntaxExtension`] - a lowered macro representation, contains its expander
  function, which transforms a [`TokenStream`] or AST into another
  [`TokenStream`] or AST + some additional data like stability, or a list of
  unstable features allowed inside the macro.
- [`SyntaxExtensionKind`] - expander functions may have several different
  signatures (take one token stream, or two, or a piece of AST, etc). This is
  an `enum` that lists them.
- [`BangProcMacro`]/[`TTMacroExpander`]/[`AttrProcMacro`]/[`MultiItemModifier`] -
  `trait`s representing the expander function signatures.

[`SyntaxExtension`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/struct.SyntaxExtension.html
[`SyntaxExtensionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/enum.SyntaxExtensionKind.html
[`BangProcMacro`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.BangProcMacro.html
[`TTMacroExpander`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.TTMacroExpander.html
[`AttrProcMacro`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.AttrProcMacro.html
[`MultiItemModifier`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/base/trait.MultiItemModifier.html

## Macros By Example

MBEs have their own parser distinct from the Rust parser. When macros are
expanded, we may invoke the MBE parser to parse and expand a macro.  The
MBE parser, in turn, may call the Rust parser when it needs to bind a
metavariable (e.g. `$my_expr`) while parsing the contents of a macro
invocation. The code for macro expansion is in
[`compiler/rustc_expand/src/mbe/`][code_dir].

### Example

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

Here `$mvar` is called a _metavariable_. Unlike normal variables, rather than
binding to a value _at runtime_, a metavariable binds _at compile time_ to a
tree of _tokens_.  A _token_ is a single "unit" of the grammar, such as an
identifier (e.g. `foo`) or punctuation (e.g. `=>`). There are also other
special tokens, such as `EOF`, which its self indicates that there are no more
tokens. There are token trees resulting from the paired parentheses-like
characters (`(`...`)`, `[`...`]`, and `{`...`}`) – they include the open and
close and all the tokens in between (Rust requires that parentheses-like
characters be balanced). Having macro expansion operate on token streams
rather than the raw bytes of a source-file abstracts away a lot of complexity.
The macro expander (and much of the rest of the compiler) doesn't consider
the exact line and column of some syntactic construct in the code; it considers
which constructs are used in the code. Using tokens allows us to care about
_what_ without worrying about _where_. For more information about tokens, see
the [Parsing][parsing] chapter of this book.

```rust,ignore
printer!(print foo); // `foo` is a variable
```

The process of expanding the macro invocation into the syntax tree
`println!("{}", foo)` and then expanding the syntax tree into a call to
`Display::fmt` is one common example of _macro expansion_.

### The MBE parser

There are two parts to MBE expansion done by the macro parser: 
  1. parsing the definition, and,
  2. parsing the invocations. 

We think of the MBE parser as a nondeterministic finite automaton (NFA) based
regex parser since it uses an algorithm similar in spirit to the [Earley
parsing algorithm](https://en.wikipedia.org/wiki/Earley_parser). The macro
parser is defined in
[`compiler/rustc_expand/src/mbe/macro_parser.rs`][code_mp].

The interface of the macro parser is as follows (this is slightly simplified):

```rust,ignore
fn parse_tt(
    &mut self,
    parser: &mut Cow<'_, Parser<'_>>,
    matcher: &[MatcherLoc]
) -> ParseResult
```

We use these items in macro parser:

- a `parser` variable is a reference to the state of a normal Rust parser,
  including the token stream and parsing session. The token stream is what we
  are about to ask the MBE parser to parse. We will consume the raw stream of
  tokens and output a binding of metavariables to corresponding token trees.
  The parsing session can be used to report parser errors.
- a `matcher` variable is a sequence of [`MatcherLoc`]s that we want to match
  the token stream against. They're converted from token trees before matching.

[`MatcherLoc`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser/enum.MatcherLoc.html

In the analogy of a regex parser, the token stream is the input and we are
matching it against the pattern defined by matcher. Using our examples, the
token stream could be the stream of tokens containing the inside of the example
invocation `print foo`, while matcher might be the sequence of token (trees)
`print $mvar:ident`.

The output of the parser is a [`ParseResult`], which indicates which of
three cases has occurred:

- **Success**: the token stream matches the given matcher and we have produced a
  binding from metavariables to the corresponding token trees.
- **Failure**: the token stream does not match matcher and results in an error
  message such as "No rule expected token ...".
- **Error**: some fatal error has occurred _in the parser_. For example, this
  happens if there is more than one pattern match, since that indicates the
  macro is ambiguous.

The full interface is defined [here][code_parse_int].

The macro parser does pretty much exactly the same as a normal regex parser
with one exception: in order to parse different types of metavariables, such as
`ident`, `block`, `expr`, etc., the macro parser must call back to the normal
Rust parser. Both the definition and invocation of macros are parsed using
the parser in a process which is non-intuitively self-referential. 

The code to parse macro _definitions_ is in
[`compiler/rustc_expand/src/mbe/macro_rules.rs`][code_mr]. It defines the
pattern for matching a macro definition as `$( $lhs:tt => $rhs:tt );+`. In
other words, a `macro_rules` definition should have in its body at least one
occurrence of a token tree followed by `=>` followed by another token tree.
When the compiler comes to a `macro_rules` definition, it uses this pattern to
match the two token trees per the rules of the definition of the macro, _thereby
utilizing the macro parser itself_. In our example definition, the
metavariable `$lhs` would match the patterns of both arms: `(print
$mvar:ident)` and `(print twice $mvar:ident)`. And `$rhs` would match the
bodies of both arms: `{ println!("{}", $mvar); }` and `{ println!("{}", $mvar);
println!("{}", $mvar); }`. The parser keeps this knowledge around for when it
needs to expand a macro invocation.

When the compiler comes to a macro invocation, it parses that invocation using
a NFA-based macro parser described above. However, the matcher variable
used is the first token tree (`$lhs`) extracted from the arms of the macro
_definition_. Using our example, we would try to match the token stream `print
foo` from the invocation against the matchers `print $mvar:ident` and `print
twice $mvar:ident` that we previously extracted from the definition. The
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
in [`compiler/rustc_expand/src/mbe/macro_parser.rs`][code_mp].

## Procedural Macros

Procedural macros are also expanded during parsing. However, rather than
having a parser in the compiler, proc macros are implemented as custom,
third-party crates. The compiler will compile the proc macro crate and
specially annotated functions in them (i.e. the proc macro itself), passing
them a stream of tokens. A proc macro can then transform the token stream and
output a new token stream, which is synthesized into the AST.

The token stream type used by proc macros is _stable_, so `rustc` does not
use it internally. The compiler's (unstable) token stream is defined in
[`rustc_ast::tokenstream::TokenStream`][rustcts]. This is converted into the
stable [`proc_macro::TokenStream`][stablets] and back in
[`rustc_expand::proc_macro`][pm] and [`rustc_expand::proc_macro_server`][pms].
Since the Rust ABI is currently unstable, we use the C ABI for this conversion.

[tsmod]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/tokenstream/index.html
[rustcts]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/tokenstream/struct.TokenStream.html
[stablets]: https://doc.rust-lang.org/proc_macro/struct.TokenStream.html
[pm]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/proc_macro/index.html
[pms]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/proc_macro_server/index.html
[`ParseResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_expand/mbe/macro_parser/enum.ParseResult.html

<!-- TODO(rylev): more here. [#1160](https://github.com/rust-lang/rustc-dev-guide/issues/1160) -->

### Custom Derive

Custom derives are a special type of proc macro.

### Macros By Example and Macros 2.0

There is an legacy and mostly undocumented effort to improve the MBE system
by giving it more hygiene-related features, better scoping and visibility
rules, etc. Internally this uses the same machinery as today's MBEs with some
additional syntactic sugar and are allowed to be in namespaces.

<!-- TODO(rylev): more? [#1160](https://github.com/rust-lang/rustc-dev-guide/issues/1160) -->
