# Macro expansion

> `libsyntax`, `librustc_expand`, and `libsyntax_ext` are all undergoing
> refactoring, so some of the links in this chapter may be broken.

Macro expansion happens during parsing. `rustc` has two parsers, in fact: the
normal Rust parser, and the macro parser. During the parsing phase, the normal
Rust parser will set aside the contents of macros and their invocations. Later,
before name resolution, macros are expanded using these portions of the code.
The macro parser, in turn, may call the normal Rust parser when it needs to
bind a metavariable (e.g.  `$my_expr`) while parsing the contents of a macro
invocation. The code for macro expansion is in
[`src/libsyntax_expand/mbe/`][code_dir]. This chapter aims to explain how macro
expansion works.

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

### The macro parser

There are two parts to macro expansion: parsing the definition and parsing the
invocations. Interestingly, both are done by the macro parser.

Basically, the macro parser is like an NFA-based regex parser. It uses an
algorithm similar in spirit to the [Earley parsing
algorithm](https://en.wikipedia.org/wiki/Earley_parser). The macro parser is
defined in [`src/libsyntax_expand/mbe/macro_parser.rs`][code_mp].

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
three cases has occurred:

- Success: `tts` matches the given matcher `ms`, and we have produced a binding
  from metavariables to the corresponding token trees.
- Failure: `tts` does not match `ms`. This results in an error message such as
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
[`src/libsyntax_expand/mbe/macro_rules.rs`][code_mr]. It defines the pattern for
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
in [`src/libsyntax_expand/mbe/macro_parser.rs`][code_mp].

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


[code_dir]: https://github.com/rust-lang/rust/tree/master/src/libsyntax_expand/mbe
[code_mp]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax_expand/mbe/macro_parser
[code_mr]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax_expand/mbe/macro_rules
[code_parse_int]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax_expand/mbe/macro_parser/fn.parse.html
[parsing]: ./the-parser.html


# Discussion about hygiene

The rest of this chapter is a dump of a discussion between `mark-i-m` and
`petrochenkov` about Macro Expansion and Hygiene. I am pasting it here so that
it never gets lost until we can make it into a proper chapter.

```txt
mark-i-m: @Vadim Petrochenkov Hi :wave:
I was wondering if you would have a chance sometime in the next month or so to
just have a zulip discussion where you tell us (WG-learning) everything you
know about macros/expansion/hygiene. We were thinking this could be less formal
(and less work for you) than compiler lecture series lecture... thoughts?

mark-i-m: The goal is to fill out that long-standing gap in the rustc-guide

Vadim Petrochenkov: Ok, I'm at UTC+03:00 and generally available in the
evenings (or weekends).

mark-i-m: @Vadim Petrochenkov Either of those works for me (your evenings are
about lunch time for me :) ) Is there a particular date that would work best
for you?

mark-i-m: @WG-learning Does anyone else have a preferred date?

    Vadim Petrochenkov:

    Is there a particular date that would work best for you?

Nah, not much difference.  (If something changes for a specific day, I'll
notify.)

Santiago Pastorino: week days are better, but I'd say let's wait for @Vadim
Petrochenkov to say when they are ready for it and we can set a date

Santiago Pastorino: also, we should record this so ... I guess it doesn't
matter that much when :)

    mark-i-m:

    also, we should record this so ... I guess it doesn't matter that much when
    :)

@Santiago Pastorino My thinking was to just use zulip, so we would have the log

mark-i-m: @Vadim Petrochenkov @WG-learning How about 2 weeks from now: July 24
at 5pm UTC time (if I did the math right, that should be evening for Vadim)

Amanjeev Sethi: i can try and do this but I am starting a new job that week so
cannot promise.

    Santiago Pastorino:

    Vadim Petrochenkov @WG-learning How about 2 weeks from now: July 24 at 5pm
    UTC time (if I did the math right, that should be evening for Vadim)

works perfect for me

Santiago Pastorino: @mark-i-m I have access to the compiler calendar so I can
add something there

Santiago Pastorino: let me know if you want to add an event to the calendar, I
can do that

Santiago Pastorino: how long it would be?

    mark-i-m:

    let me know if you want to add an event to the calendar, I can do that

mark-i-m: That could be good :+1:

    mark-i-m:

    how long it would be?

Let's start with 30 minutes, and if we need to schedule another we cna

    Vadim Petrochenkov:

    5pm UTC

1-2 hours later would be better, 5pm UTC is not evening enough.

Vadim Petrochenkov: How exactly do you plan the meeting to go (aka how much do
I need to prepare)?

    Santiago Pastorino:

        5pm UTC

    1-2 hours later would be better, 5pm UTC is not evening enough.

Scheduled for 7pm UTC then

    Santiago Pastorino:

    How exactly do you plan the meeting to go (aka how much do I need to
    prepare)?

/cc @mark-i-m

mark-i-m: @Vadim Petrochenkov

    How exactly do you plan the meeting to go (aka how much do I need to
    prepare)?

My hope was that this could be less formal than for a compiler lecture series,
but it would be nice if you could have in your mind a tour of the design and
the code

That is, imagine that a new person was joining the compiler team and needed to
get up to speed about macros/expansion/hygiene. What would you tell such a
person?

mark-i-m: @Vadim Petrochenkov Are we still on for tomorrow at 7pm UTC?

Vadim Petrochenkov: Yes.

Santiago Pastorino: @Vadim Petrochenkov @mark-i-m I've added an event on rust
compiler team calendar

mark-i-m: @WG-learning @Vadim Petrochenkov Hello!

mark-i-m: We will be starting in ~7 minutes

mark-i-m: :wave:

Vadim Petrochenkov: I'm here.

mark-i-m: Cool :)

Santiago Pastorino: hello @Vadim Petrochenkov

mark-i-m: Shall we start?

mark-i-m: First off, @Vadim Petrochenkov Thanks for doing this!

Vadim Petrochenkov: Here's some preliminary data I prepared.

Vadim Petrochenkov: Below I'll assume #62771 and #62086 has landed.

Vadim Petrochenkov: Where to find the code: libsyntax_pos/hygiene.rs -
structures related to hygiene and expansion that are kept in global data (can
be accessed from any Ident without any context) libsyntax_pos/lib.rs - some
secondary methods like macro backtrace using primary methods from hygiene.rs
libsyntax_ext - implementations of built-in macros (including macro attributes
and derives) and some other early code generation facilities like injection of
standard library imports or generation of test harness.  libsyntax/config.rs -
implementation of cfg/cfg_attr (they treated specially from other macros),
should probably be moved into libsyntax/ext.  libsyntax/tokenstream.rs +
libsyntax/parse/token.rs - structures for compiler-side tokens, token trees,
and token streams.  libsyntax/ext - various expansion-related stuff
libsyntax/ext/base.rs - basic structures used by expansion
libsyntax/ext/expand.rs - some expansion structures and the bulk of expansion
infrastructure code - collecting macro invocations, calling into resolve for
them, calling their expanding functions, and integrating the results back into
AST libsyntax/ext/placeholder.rs - the part of expand.rs responsible for
"integrating the results back into AST" basicallly, "placeholder" is a
temporary AST node replaced with macro expansion result nodes
libsyntax/ext/builer.rs - helper functions for building AST for built-in macros
in libsyntax_ext (and user-defined syntactic plugins previously), can probably
be moved into libsyntax_ext these days libsyntax/ext/proc_macro.rs +
libsyntax/ext/proc_macro_server.rs - interfaces between the compiler and the
stable proc_macro library, converting tokens and token streams between the two
representations and sending them through C ABI libsyntax/ext/tt -
implementation of macro_rules, turns macro_rules DSL into something with
signature Fn(TokenStream) -> TokenStream that can eat and produce tokens,
@mark-i-m knows more about this librustc_resolve/macros.rs - resolving macro
paths, validating those resolutions, reporting various "not found"/"found, but
it's unstable"/"expected x, found y" errors librustc/hir/map/def_collector.rs +
librustc_resolve/build_reduced_graph.rs - integrate an AST fragment freshly
expanded from a macro into various parent/child structures like module
hierarchy or "definition paths"

Primary structures: HygieneData - global piece of data containing hygiene and
expansion info that can be accessed from any Ident without any context ExpnId -
ID of a macro call or desugaring (and also expansion of that call/desugaring,
depending on context) ExpnInfo/InternalExpnData - a subset of properties from
both macro definition and macro call available through global data
SyntaxContext - ID of a chain of nested macro definitions (identified by
ExpnIds) SyntaxContextData - data associated with the given SyntaxContext,
mostly a cache for results of filtering that chain in different ways Span - a
code location + SyntaxContext Ident - interned string (Symbol) + Span, i.e. a
string with attached hygiene data TokenStream - a collection of TokenTrees
TokenTree - a token (punctuation, identifier, or literal) or a delimited group
(anything inside ()/[]/{}) SyntaxExtension - a lowered macro representation,
contains its expander function transforming a tokenstream or AST into
tokenstream or AST + some additional data like stability, or a list of unstable
features allowed inside the macro.  SyntaxExtensionKind - expander functions
may have several different signatures (take one token stream, or two, or a
piece of AST, etc), this is an enum that lists them
ProcMacro/TTMacroExpander/AttrProcMacro/MultiItemModifier - traits representing
the expander signatures (TODO: change and rename the signatures into something
more consistent) trait Resolver - a trait used to break crate dependencies (so
resolver services can be used in libsyntax, despite librustc_resolve and pretty
much everything else depending on libsyntax) ExtCtxt/ExpansionData - various
intermediate data kept and used by expansion infra in the process of its work
AstFragment - a piece of AST that can be produced by a macro (may include
multiple homogeneous AST nodes, like e.g. a list of items) Annotatable - a
piece of AST that can be an attribute target, almost same thing as AstFragment
except for types and patterns that can be produced by macros but cannot be
annotated with attributes (TODO: Merge into AstFragment) trait MacResult - a
"polymorphic" AST fragment, something that can turn into a different
AstFragment depending on its context (aka AstFragmentKind - item, or
expression, or pattern etc.) Invocation/InvocationKind - a structure describing
a macro call, these structures are collected by the expansion infra
(InvocationCollector), queued, resolved, expanded when resolved, etc.

Primary algorithms / actions: TODO

mark-i-m: Very useful :+1:

mark-i-m: @Vadim Petrochenkov Zulip doesn't have an indication of typing, so
I'm not sure if you are waiting for me or not

Vadim Petrochenkov: The TODO part should be about how a crate transitions from
the state "macros exist as written in source" to "all macros are expanded", but
I didn't write it yet.

Vadim Petrochenkov: (That should probably better happen off-line.)

Vadim Petrochenkov: Now, if you have any questions?

mark-i-m: Thanks :)

mark-i-m: /me is still reading :P

mark-i-m: Ok

mark-i-m: So I guess my first question is about hygiene, since that remains the
most mysterious to me... My understanding is that the parser outputs AST nodes,
where each node has a Span

mark-i-m: In the absence of macros and desugaring, what does the syntax context
of an AST node look like?

mark-i-m: @Vadim Petrochenkov

Vadim Petrochenkov: Not each node, but many of them.  When a node is not
macro-expanded, its context is 0.

Vadim Petrochenkov: aka SyntaxContext::empty()

Vadim Petrochenkov: it's a chain that consists of one expansion - expansion 0
aka ExpnId::root.

mark-i-m: Do all expansions start at root?

Vadim Petrochenkov: Also, SyntaxContext:empty() is its own father.

mark-i-m: Is this actually stored somewhere or is it a logical value?

Vadim Petrochenkov: All expansion hyerarchies (there are several of them) start
at ExpnId::root.

Vadim Petrochenkov: Vectors in HygieneData has entries for both ctxt == 0 and
expn_id == 0.

Vadim Petrochenkov: I don't think anyone looks into them much though.

mark-i-m: Ok

Vadim Petrochenkov: Speaking of multiple hierarchies...

mark-i-m: Go ahead :)

Vadim Petrochenkov: One is parent (expn_id1) -> parent(expn_id2) -> ...

Vadim Petrochenkov: This is the order in which macros are expanded.

Vadim Petrochenkov: Well.

Vadim Petrochenkov: When we are expanding one macro another macro is revealed
in its output.

Vadim Petrochenkov: That's the parent-child relation in this hierarchy.

Vadim Petrochenkov: InternalExpnData::parent is the child->parent link.

mark-i-m: So in the above chain expn_id1 is the child?

Vadim Petrochenkov: Yes.

Vadim Petrochenkov: The second one is parent (SyntaxContext1) ->
parent(SyntaxContext2) -> ...

Vadim Petrochenkov: This is about nested macro definitions.  When we are
expanding one macro another macro definition is revealed in its output.

Vadim Petrochenkov: SyntaxContextData::parent is the child->parent link here.

Vadim Petrochenkov: So, SyntaxContext is the whole chain in this hierarchy, and
outer_expns are individual elements in the chain.

mark-i-m: So for example, suppose I have the following:

macro_rules! foo { () => { println!(); } }

fn main() { foo!(); }

Then AST nodes that are finally generated would have parent(expn_id_println) ->
parent(expn_id_foo), right?

Vadim Petrochenkov: Pretty common construction (at least it was, before
refactorings) is SyntaxContext::empty().apply_mark(expn_id), which means...

    Vadim Petrochenkov:

    Then AST nodes that are finally generated would have
    parent(expn_id_println) -> parent(expn_id_foo), right?

Yes.

    mark-i-m:

    and outer_expns are individual elements in the chain.

Sorry, what is outer_expns?

Vadim Petrochenkov: SyntaxContextData::outer_expn

mark-i-m: Thanks :) Please continue

Vadim Petrochenkov: ...which means a token produced by a built-in macro (which
is defined in the root effectively).

mark-i-m: Where does the expn_id come from?

Vadim Petrochenkov: Or a stable proc macro, which are always considered to be
defined in the root because they are always cross-crate, and we don't have the
cross-crate hygiene implemented, ha-ha.

    Vadim Petrochenkov:

    Where does the expn_id come from?

Vadim Petrochenkov: ID of the built-in macro call like line!().

Vadim Petrochenkov: Assigned continuously from 0 to N as soon as we discover
new macro calls.

mark-i-m: Sorry, I didn't quite understand. Do you mean that only built-in
macros receive continuous IDs?

Vadim Petrochenkov: So, the second hierarchy has a catch - the context
transplantation hack -
https://github.com/rust-lang/rust/pull/51762#issuecomment-401400732.

    Vadim Petrochenkov:

    Do you mean that only built-in macros receive continuous IDs?

Vadim Petrochenkov: No, all macro calls receive ID.

Vadim Petrochenkov: Built-ins have the typical pattern
SyntaxContext::empty().apply_mark(expn_id) for syntax contexts produced by
them.

mark-i-m: I see, but this pattern is only used for built-ins, right?

Vadim Petrochenkov: And also all stable proc macros, see the comments above.

mark-i-m: Got it

Vadim Petrochenkov: The third hierarchy is call-site hierarchy.

Vadim Petrochenkov: If foo!(bar!(ident)) expands into ident

Vadim Petrochenkov: then hierarchy 1 is root -> foo -> bar -> ident

Vadim Petrochenkov: but hierarchy 3 is root -> ident

Vadim Petrochenkov: ExpnInfo::call_site is the child-parent link in this case.

mark-i-m: When we expand, do we expand foo first or bar? Why is there a
hierarchy 1 here? Is that foo expands first and it expands to something that
contains bar!(ident)?

Vadim Petrochenkov: Ah, yes, let's assume both foo and bar are identity macros.

Vadim Petrochenkov: Then foo!(bar!(ident)) -> expand -> bar!(ident) -> expand
-> ident

Vadim Petrochenkov: If bar were expanded first, that would be eager expansion -
https://github.com/rust-lang/rfcs/pull/2320.

mark-i-m: And after we expand only foo! presumably whatever intermediate state
has heirarchy 1 of root->foo->(bar_ident), right?

Vadim Petrochenkov: (We have it hacked into some built-in macros, but not
generally.)

    Vadim Petrochenkov:

    And after we expand only foo! presumably whatever intermediate state has
    heirarchy 1 of root->foo->(bar_ident), right?

Vadim Petrochenkov: Yes.

mark-i-m: Got it :)

mark-i-m: It looks like we have ~5 minutes left. This has been very helpful
already, but I also have more questions. Shall we try to schedule another
meeting in the future?

Vadim Petrochenkov: Sure, why not.

Vadim Petrochenkov: A thread for offline questions-answers would be good too.

    mark-i-m:

    A thread for offline questions-answers would be good too.

I don't mind using this thread, since it already has a lot of info in it. We
also plan to summarize the info from this thread into the rustc-guide.

    Sure, why not.

Unfortunately, I'm unavailable for a few weeks. Would August 21-ish work for
you (and @WG-learning )?

mark-i-m: @Vadim Petrochenkov Thanks very much for your time and knowledge!

mark-i-m: One last question: are there more hierarchies?

Vadim Petrochenkov: Not that I know of.  Three + the context transplantation
hack is already more complex than I'd like.

mark-i-m: Yes, one wonders what it would be like if one also had to think about
eager expansion...

Santiago Pastorino: sorry but I couldn't follow that much today, will read it
when I have some time later

Santiago Pastorino: btw https://github.com/rust-lang/rustc-guide/issues/398

mark-i-m: @Vadim Petrochenkov Would 7pm UTC on August 21 work for a followup?

Vadim Petrochenkov: Tentatively yes.

mark-i-m: @Vadim Petrochenkov @WG-learning Does this still work for everyone?

Vadim Petrochenkov: August 21 is still ok.

mark-i-m: @WG-learning @Vadim Petrochenkov We will start in ~30min

Vadim Petrochenkov: Oh.  Thanks for the reminder, I forgot about this entirely.

mark-i-m: Hello!

Vadim Petrochenkov: (I'll be here in a couple of minutes.)

Vadim Petrochenkov: Ok, I'm here.

mark-i-m: Hi :)

Vadim Petrochenkov: Hi.

mark-i-m: so last time, we talked about the 3 context heirarchies

Vadim Petrochenkov: Right.

mark-i-m: Was there anything you wanted to add to that? If not, I think it
would be good to get a big-picture... Given some piece of rust code, how do we
get to the point where things are expanded and hygiene context is computed?

mark-i-m: (I'm assuming that hygiene info is computed as we expand stuff, since
I don't think you can discover it beforehand)

Vadim Petrochenkov: Ok, let's move from hygiene to expansion.

Vadim Petrochenkov: Especially given that I don't remember the specific hygiene
algorithms like adjust in detail.

    Vadim Petrochenkov:

    Given some piece of rust code, how do we get to the point where things are
    expanded

So, first of all, the "some piece of rust code" is the whole crate.

mark-i-m: Just to confirm, the algorithms are well-encapsulated, right? Like a
function or a struct as opposed to a bunch of conventions distributed across
the codebase?

Vadim Petrochenkov: We run fully_expand_fragment in it.

    Vadim Petrochenkov:

    Just to confirm, the algorithms are well-encapsulated, right?

Yes, the algorithmic parts are entirely inside hygiene.rs.

Vadim Petrochenkov: Ok, some are in fn resolve_crate_root, but those are hacks.

Vadim Petrochenkov: (Continuing about expansion.) If fully_expand_fragment is
run not on a whole crate, it means that we are performing eager expansion.

Vadim Petrochenkov: Eager expansion is done for arguments of some built-in
macros that expect literals.

Vadim Petrochenkov: It generally performs a subset of actions performed by the
non-eager expansion.

Vadim Petrochenkov: So, I'll talk about non-eager expansion for now.

mark-i-m: Eager expansion is not exposed as a language feature, right? i.e. it
is not possible for me to write an eager macro?

Vadim Petrochenkov:
https://github.com/rust-lang/rust/pull/53778#issuecomment-419224049 (vvv The
link is explained below vvv )

    Vadim Petrochenkov:

    Eager expansion is not exposed as a language feature, right? i.e. it is not
    possible for me to write an eager macro?

Yes, it's entirely an ability of some built-in macros.

Vadim Petrochenkov: Not exposed for general use.

Vadim Petrochenkov: fully_expand_fragment works in iterations.

Vadim Petrochenkov: Iterations looks roughly like this:
- Resolve imports in our partially built crate as much as possible.
- Collect as many macro invocations as possible from our partially built crate
  (fn-like, attributes, derives) from the crate and add them to the queue.

    Vadim Petrochenkov: Take a macro from the queue, and attempt to resolve it.

    Vadim Petrochenkov: If it's resolved - run its expander function that
    consumes tokens or AST and produces tokens or AST (depending on the macro
    kind).

    Vadim Petrochenkov: (If it's not resolved, then put it back into the
    queue.)

Vadim Petrochenkov: ^^^ That's where we fill in the hygiene data associated
with ExpnIds.

mark-i-m: When we put it back in the queue?

mark-i-m: or do you mean the collect step in general?

Vadim Petrochenkov: Once we resolved the macro call to the macro definition we
know everything about the macro and can call set_expn_data to fill in its
properties in the global data.

Vadim Petrochenkov: I mean, immediately after successful resolution.

Vadim Petrochenkov: That's the first part of hygiene data, the second one is
associated with SyntaxContext rather than with ExpnId, it's filled in later
during expansion.

Vadim Petrochenkov: So, after we run the macro's expander function and got a
piece of AST (or got tokens and parsed them into a piece of AST) we need to
integrate that piece of AST into the big existing partially built AST.

Vadim Petrochenkov: This integration is a really important step where the next
things happen:
- NodeIds are assigned.

    Vadim Petrochenkov: "def paths"s and their IDs (DefIds) are created

    Vadim Petrochenkov: Names are put into modules from the resolver point of
    view.

Vadim Petrochenkov: So, we are basically turning some vague token-like mass
into proper set in stone hierarhical AST and side tables.

Vadim Petrochenkov: Where exactly this happens - NodeIds are assigned by
InvocationCollector (which also collects new macro calls from this new AST
piece and adds them to the queue), DefIds are created by DefCollector, and
modules are filled by BuildReducedGraphVisitor.

Vadim Petrochenkov: These three passes run one after another on every AST
fragment freshly expanded from a macro.

Vadim Petrochenkov: After expanding a single macro and integrating its output
we again try to resolve all imports in the crate, and then return to the big
queue processing loop and pick up the next macro.

Vadim Petrochenkov: Repeat until there's no more macros.  Vadim Petrochenkov:

mark-i-m: The integration step is where we would get parser errors too right?

mark-i-m: Also, when do we know definitively that resolution has failed for
particular ident?

    Vadim Petrochenkov:

    The integration step is where we would get parser errors too right?

Yes, if the macro produced tokens (rather than AST directly) and we had to
parse them.

    Vadim Petrochenkov:

    when do we know definitively that resolution has failed for particular
    ident?

So, ident is looked up in a number of scopes during resolution.  From closest
like the current block or module, to far away like preludes or built-in types.

Vadim Petrochenkov: If lookup is certainly failed in all of the scopes, then
it's certainly failed.

mark-i-m: This is after all expansions and integrations are done, right?

Vadim Petrochenkov: "Certainly" is determined differently for different scopes,
e.g. for a module scope it means no unexpanded macros and no unresolved glob
imports in that module.

    Vadim Petrochenkov:

    This is after all expansions and integrations are done, right?

For macro and import names this happens during expansions and integrations.

mark-i-m: Makes sense

Vadim Petrochenkov: For all other names we certainly know whether a name is
resolved successfully or not on the first attempt, because no new names can
appear.

Vadim Petrochenkov: (They are resolved in a later pass, see
librustc_resolve/late.rs.)

mark-i-m: And if at the end of the iteration, there are still things in the
queue that can't be resolve, this represents an error, right?

mark-i-m: i.e. an undefined macro?

Vadim Petrochenkov: Yes, if we make no progress during an iteration, then we
are stuck and that state represent an error.

Vadim Petrochenkov: We attempt to recover though, using dummies expanding into
nothing or ExprKind::Err or something like that for unresolved macros.

mark-i-m: This is for the purposes of diagnostics, though, right?

Vadim Petrochenkov: But if we are going through recovery, then compilation must
result in an error anyway.

Vadim Petrochenkov: Yes, that's for diagnostics, without recovery we would
stuck at the first unresolved macro or import.  Vadim Petrochenkov:

So, about the SyntaxContext hygiene...

Vadim Petrochenkov: New syntax contexts are created during macro expansion.

Vadim Petrochenkov: If the token had context X before being produced by a
macro, e.g. here ident has context SyntaxContext::root(): Vadim Petrochenkov:

macro m() { ident }

Vadim Petrochenkov: , then after being produced by the macro it has context X
-> macro_id.

Vadim Petrochenkov: I.e. our ident has context ROOT -> id(m) after it's
produced by m.

Vadim Petrochenkov: The "chaining operator" -> is apply_mark in compiler code.
Vadim Petrochenkov:

macro m() { macro n() { ident } }

Vadim Petrochenkov: In this example the ident has context ROOT originally, then
ROOT -> id(m), then ROOT -> id(m) -> id(n).

Vadim Petrochenkov: Note that these chains are not entirely determined by their
last element, in other words ExpnId is not isomorphic to SyntaxCtxt.

Vadim Petrochenkov: Couterexample: Vadim Petrochenkov:

macro m($i: ident) { macro n() { ($i, bar) } }

m!(foo);

Vadim Petrochenkov: foo has context ROOT -> id(n) and bar has context ROOT ->
id(m) -> id(n) after all the expansions.

mark-i-m: Cool :)

mark-i-m: It looks like we are out of time

mark-i-m: Is there anything you wanted to add?

mark-i-m: We can schedule another meeting if you would like

Vadim Petrochenkov: Yep, 23.06 already.  No, I think this is an ok point to
stop.

mark-i-m: :+1:

mark-i-m: Thanks @Vadim Petrochenkov ! This was very helpful

Vadim Petrochenkov: Yeah, we can schedule another one.  So far it's been like 1
hour of meetings per month? Certainly not a big burden.
```
