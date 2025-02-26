# Guide to rust-analyzer

## About the guide

This guide describes the current state of rust-analyzer as of the 2024-01-01 release
(git tag [2024-01-01]). Its purpose is to document various problems and
architectural solutions related to the problem of building IDE-first compiler
for Rust. There is a video version of this guide as well -
however, it's based on an older 2019-01-20 release (git tag [guide-2019-01]):
https://youtu.be/ANKBNiSWyfc.

[guide-2019-01]: https://github.com/rust-lang/rust-analyzer/tree/guide-2019-01
[2024-01-01]: https://github.com/rust-lang/rust-analyzer/tree/2024-01-01

<!-- toc -->

## The big picture

On the highest possible level, rust-analyzer is a stateful component. A client may
apply changes to the analyzer (new contents of `foo.rs` file is "fn main() {}")
and it may ask semantic questions about the current state (what is the
definition of the identifier with offset 92 in file `bar.rs`?). Two important
properties hold:

* Analyzer does not do any I/O. It starts in an empty state and all input data is
  provided via `apply_change` API.

* Only queries about the current state are supported. One can, of course,
  simulate undo and redo by keeping a log of changes and inverse changes respectively.

## IDE API

To see the bigger picture of how the IDE features work, let's take a look at the [`AnalysisHost`] and
[`Analysis`] pair of types. `AnalysisHost` has three methods:

* `default()` for creating an empty analysis instance
* `apply_change(&mut self)` to make changes (this is how you get from an empty
  state to something interesting)
* `analysis(&self)` to get an instance of `Analysis`

`Analysis` has a ton of methods for IDEs, like `goto_definition`, or
`completions`. Both inputs and outputs of `Analysis`' methods are formulated in
terms of files and offsets, and **not** in terms of Rust concepts like structs,
traits, etc. The "typed" API with Rust specific types is slightly lower in the
stack, we'll talk about it later.

[`AnalysisHost`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide/src/lib.rs#L161-L213
[`Analysis`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide/src/lib.rs#L220-L761

The reason for this separation of `Analysis` and `AnalysisHost` is that we want to apply
changes "uniquely", but we might also want to fork an `Analysis` and send it to
another thread for background processing. That is, there is only a single
`AnalysisHost`, but there may be several (equivalent) `Analysis`.

Note that all of the `Analysis` API return `Cancellable<T>`. This is required to
be responsive in an IDE setting. Sometimes a long-running query is being computed
and the user types something in the editor and asks for completion. In this
case, we cancel the long-running computation (so it returns `Err(Cancelled)`),
apply the change and execute request for completion. We never use stale data to
answer requests. Under the cover, `AnalysisHost` "remembers" all outstanding
`Analysis` instances. The `AnalysisHost::apply_change` method cancels all
`Analysis`es, blocks until all of them are `Dropped` and then applies changes
in-place. This may be familiar to Rustaceans who use read-write locks for interior
mutability.

Next, let's talk about what the inputs to the `Analysis` are, precisely.

## Inputs

rust-analyzer never does any I/O itself, all inputs get passed explicitly via
the `AnalysisHost::apply_change` method, which accepts a single argument, a
`Change`. [`Change`] is a wrapper for `FileChange` that adds proc-macro knowledge.
[`FileChange`] is a builder for a single change "transaction", so it suffices
to study its methods to understand all the input data.

[`Change`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-expand/src/change.rs#L10-L42
[`FileChange`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/base-db/src/change.rs#L14-L78

The `change_file` method controls the set of the input files, where each file
has an integer id (`FileId`, picked by the client) and text (`Option<Arc<str>>`).
Paths are tricky; they'll be explained below, in source roots section,
together with the `set_roots` method. The "source root" [`is_library`] flag
along with the concept of [`durability`] allows us to add a group of files which
are assumed to rarely change. It's mostly an optimization and does not change
the fundamental picture.

[`is_library`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/base-db/src/input.rs#L38
[`durability`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/base-db/src/change.rs#L80-L86

The `set_crate_graph` method allows us to control how the input files are partitioned
into compilation units -- crates. It also controls (in theory, not implemented
yet) `cfg` flags. `CrateGraph` is a directed acyclic graph of crates. Each crate
has a root `FileId`, a set of active `cfg` flags and a set of dependencies. Each
dependency is a pair of a crate and a name. It is possible to have two crates
with the same root `FileId` but different `cfg`-flags/dependencies. This model
is lower than Cargo's model of packages: each Cargo package consists of several
targets, each of which is a separate crate (or several crates, if you try
different feature combinations).

Procedural macros are inputs as well, roughly modeled as a crate with a bunch of
additional black box `dyn Fn(TokenStream) -> TokenStream` functions.

Soon we'll talk how we build an LSP server on top of `Analysis`, but first,
let's deal with that paths issue.

## Source roots (a.k.a. "Filesystems are horrible")

This is a non-essential section, feel free to skip.

The previous section said that the filesystem path is an attribute of a file,
but this is not the whole truth. Making it an absolute `PathBuf` will be bad for
several reasons. First, filesystems are full of (platform-dependent) edge cases:

* It's hard (requires a syscall) to decide if two paths are equivalent.
* Some filesystems are case-sensitive (e.g. macOS).
* Paths are not necessarily UTF-8.
* Symlinks can form cycles.

Second, this might hurt the reproducibility and hermeticity of builds. In theory,
moving a project from `/foo/bar/my-project` to `/spam/eggs/my-project` should
not change a bit in the output. However, if the absolute path is a part of the
input, it is at least in theory observable, and *could* affect the output.

Yet another problem is that we really *really* want to avoid doing I/O, but with
Rust the set of "input" files is not necessarily known up-front. In theory, you
can have `#[path="/dev/random"] mod foo;`.

To solve (or explicitly refuse to solve) these problems rust-analyzer uses the
concept of a "source root". Roughly speaking, source roots are the contents of a
directory on a file system, like `/home/matklad/projects/rustraytracer/**.rs`.

More precisely, all files (`FileId`s) are partitioned into disjoint
`SourceRoot`s. Each file has a relative UTF-8 path within the `SourceRoot`.
`SourceRoot` has an identity (integer ID). Crucially, the root path of the
source root itself is unknown to the analyzer: A client is supposed to maintain a
mapping between `SourceRoot` IDs (which are assigned by the client) and actual
`PathBuf`s. `SourceRoot`s give a sane tree model of the file system to the
analyzer.

Note that `mod`, `#[path]` and `include!()` can only reference files from the
same source root. It is of course possible to explicitly add extra files to
the source root, even `/dev/random`.

## Language Server Protocol

Now let's see how the `Analysis` API is exposed via the JSON RPC based language server protocol.
The hard part here is managing changes (which can come either from the file system
or from the editor) and concurrency (we want to spawn background jobs for things
like syntax highlighting). We use the event loop pattern to manage the zoo, and
the loop is the [`GlobalState::run`] function initiated by [`main_loop`] after
[`GlobalState::new`] does a one-time initialization and tearing down of the resources.

[`main_loop`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L31-L54
[`GlobalState::new`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/global_state.rs#L148-L215
[`GlobalState::run`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L114-L140

Let's walk through a typical analyzer session!

First, we need to figure out what to analyze. To do this, we run `cargo
metadata` to learn about Cargo packages for current workspace and dependencies,
and we run `rustc --print sysroot` and scan the "sysroot"
(the directory containing the current Rust toolchain's files) to learn about crates
like `std`. This happens in the [`GlobalState::fetch_workspaces`] method.
We load this configuration at the start of the server in [`GlobalState::new`],
but it's also triggered by workspace change events and requests to reload the
workspace from the client.

[`GlobalState::fetch_workspaces`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/reload.rs#L186-L257

The [`ProjectModel`] we get after this step is very Cargo and sysroot specific,
it needs to be lowered to get the input in the form of `Change`. This happens
in [`GlobalState::process_changes`] method. Specifically

* Create `SourceRoot`s for each Cargo package(s) and sysroot.
* Schedule a filesystem scan of the roots.
* Create an analyzer's `Crate` for each Cargo **target** and sysroot crate.
* Setup dependencies between the crates.

[`ProjectModel`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/project-model/src/workspace.rs#L57-L100
[`GlobalState::process_changes`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/global_state.rs#L217-L356

The results of the scan (which may take a while) will be processed in the body
of the main loop, just like any other change. Here's where we handle:

* [File system changes](https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L273)
* [Changes from the editor](https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L801-L803)

After a single loop's turn, we group the changes into one `Change` and
[apply] it. This always happens on the main thread and blocks the loop.

[apply]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/global_state.rs#L333

To handle requests, like ["goto definition"], we create an instance of the
`Analysis` and [`schedule`] the task (which consumes `Analysis`) on the
threadpool. [The task] calls the corresponding `Analysis` method, while
massaging the types into the LSP representation. Keep in mind that if we are
executing "goto definition" on the threadpool and a new change comes in, the
task will be canceled as soon as the main loop calls `apply_change` on the
`AnalysisHost`.

["goto definition"]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L767
[`schedule`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/dispatch.rs#L138
[The task]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/handlers/request.rs#L610-L623

This concludes the overview of the analyzer's programing *interface*. Next, let's
dig into the implementation!

## Salsa

The most straightforward way to implement an "apply change, get analysis, repeat"
API would be to maintain the input state and to compute all possible analysis
information from scratch after every change. This works, but scales poorly with
the size of the project. To make this fast, we need to take advantage of the
fact that most of the changes are small, and that analysis results are unlikely
to change significantly between invocations.

To do this we use [salsa]: a framework for incremental on-demand computation.
You can skip the rest of the section if you are familiar with `rustc`'s red-green
algorithm (which is used for incremental compilation).

[salsa]: https://github.com/salsa-rs/salsa

It's better to refer to salsa's docs to learn about it. Here's a small excerpt:

The key idea of salsa is that you define your program as a set of queries. Every
query is used like a function `K -> V` that maps from some key of type `K` to a value
of type `V`. Queries come in two basic varieties:

* **Inputs**: the base inputs to your system. You can change these whenever you
  like.

* **Functions**: pure functions (no side effects) that transform your inputs
  into other values. The results of queries are memoized to avoid recomputing
  them a lot. When you make changes to the inputs, we'll figure out (fairly
  intelligently) when we can re-use these memoized values and when we have to
  recompute them.

For further discussion, its important to understand one bit of "fairly
intelligently". Suppose we have two functions, `f1` and `f2`, and one input,
`z`. We call `f1(X)` which in turn calls `f2(Y)` which inspects `i(Z)`. `i(Z)`
returns some value `V1`, `f2` uses that and returns `R1`, `f1` uses that and
returns `O`. Now, let's change `i` at `Z` to `V2` from `V1` and try to compute
`f1(X)` again. Because `f1(X)` (transitively) depends on `i(Z)`, we can't just
reuse its value as is. However, if `f2(Y)` is *still* equal to `R1` (despite
`i`'s change), we, in fact, *can* reuse `O` as result of `f1(X)`. And that's how
salsa works: it recomputes results in *reverse* order, starting from inputs and
progressing towards outputs, stopping as soon as it sees an intermediate value
that hasn't changed. If this sounds confusing to you, don't worry: it is
confusing. This illustration by @killercup might help:

<img alt="step 1" src="https://user-images.githubusercontent.com/1711539/51460907-c5484780-1d6d-11e9-9cd2-d6f62bd746e0.png" width="50%">

<img alt="step 2" src="https://user-images.githubusercontent.com/1711539/51460915-c9746500-1d6d-11e9-9a77-27d33a0c51b5.png" width="50%">

<img alt="step 3" src="https://user-images.githubusercontent.com/1711539/51460920-cda08280-1d6d-11e9-8d96-a782aa57a4d4.png" width="50%">

<img alt="step 4" src="https://user-images.githubusercontent.com/1711539/51460927-d1340980-1d6d-11e9-851e-13c149d5c406.png" width="50%">

## Salsa Input Queries

All analyzer information is stored in a salsa database. `Analysis` and
`AnalysisHost` types are essentially newtype wrappers for [`RootDatabase`]
-- a salsa database.

[`RootDatabase`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-db/src/lib.rs#L69-L324

Salsa input queries are defined in [`SourceDatabase`] and [`SourceDatabaseExt`]
(which are a part of `RootDatabase`). They closely mirror the familiar `Change`
structure: indeed, what `apply_change` does is it sets the values of input queries.

[`SourceDatabase`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/base-db/src/lib.rs#L58-L65
[`SourceDatabaseExt`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/base-db/src/lib.rs#L76-L88

## From text to semantic model

The bulk of the rust-analyzer is transforming input text into a semantic model of
Rust code: a web of entities like modules, structs, functions and traits.

An important fact to realize is that (unlike most other languages like C# or
Java) there is not a one-to-one mapping between the source code and the semantic model. A
single function definition in the source code might result in several semantic
functions: for example, the same source file might get included as a module in
several crates or a single crate might be present in the compilation DAG
several times, with different sets of `cfg`s enabled. The IDE-specific task of
mapping source code into a semantic model is inherently imprecise for
this reason and gets handled by the [`source_analyzer`].

[`source_analyzer`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir/src/source_analyzer.rs

The semantic interface is declared in the [`semantics`] module. Each entity is
identified by an integer ID and has a bunch of methods which take a salsa database
as an argument and returns other entities (which are also IDs). Internally, these
methods invoke various queries on the database to build the model on demand.
Here's [the list of queries].

[`semantics`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir/src/semantics.rs
[the list of queries]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-ty/src/db.rs#L29-L275

The first step of building the model is parsing the source code.

## Syntax trees

An important property of the Rust language is that each file can be parsed in
isolation. Unlike, say, `C++`, an `include` can't change the meaning of the
syntax. For this reason, rust-analyzer can build a syntax tree for each "source
file", which could then be reused by several semantic models if this file
happens to be a part of several crates.

The representation of syntax trees that rust-analyzer uses is similar to that of `Roslyn`
and Swift's new [libsyntax]. Swift's docs give an excellent overview of the
approach, so I skip this part here and instead outline the main characteristics
of the syntax trees:

* Syntax trees are fully lossless. Converting **any** text to a syntax tree and
  back is a total identity function. All whitespace and comments are explicitly
  represented in the tree.

* Syntax nodes have generic `(next|previous)_sibling`, `parent`,
  `(first|last)_child` functions. You can get from any one node to any other
  node in the file using only these functions.

* Syntax nodes know their range (start offset and length) in the file.

* Syntax nodes share the ownership of their syntax tree: if you keep a reference
  to a single function, the whole enclosing file is alive.

* Syntax trees are immutable and the cost of replacing the subtree is
  proportional to the depth of the subtree. Read Swift's docs to learn how
  immutable + parent pointers + cheap modification is possible.

* Syntax trees are build on best-effort basis. All accessor methods return
  `Option`s. The tree for `fn foo` will contain a function declaration with
  `None` for parameter list and body.

* Syntax trees do not know the file they are built from, they only know about
  the text.

The implementation is based on the generic [rowan] crate on top of which a
[rust-specific] AST is generated.

[libsyntax]: https://github.com/apple/swift/tree/5e2c815edfd758f9b1309ce07bfc01c4bc20ec23/lib/Syntax
[rowan]: https://github.com/rust-analyzer/rowan/tree/100a36dc820eb393b74abe0d20ddf99077b61f88
[rust-specific]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/syntax/src/ast/generated.rs

The next step in constructing the semantic model is ...

## Building a Module Tree

The algorithm for building a tree of modules is to start with a crate root
(remember, each `Crate` from a `CrateGraph` has a `FileId`), collect all `mod`
declarations and recursively process child modules. This is handled by the
[`crate_def_map_query`], with two slight variations.

[`crate_def_map_query`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/nameres.rs#L307-L324

First, rust-analyzer builds a module tree for all crates in a source root
simultaneously. The main reason for this is historical (`module_tree` predates
`CrateGraph`), but this approach also enables accounting for files which are not
part of any crate. That is, if you create a file but do not include it as a
submodule anywhere, you still get semantic completion, and you get a warning
about a free-floating module (the actual warning is not implemented yet).

The second difference is that `crate_def_map_query` does not *directly* depend on
the `SourceDatabase::parse` query. Why would calling the parse directly be bad?
Suppose the user changes the file slightly, by adding an insignificant whitespace.
Adding whitespace changes the parse tree (because it includes whitespace),
and that means recomputing the whole module tree.

We deal with this problem by introducing an intermediate [`block_def_map_query`].
This query processes the syntax tree and extracts a set of declared submodule
names. Now, changing the whitespace results in `block_def_map_query` being
re-executed for a *single* module, but because the result of this query stays
the same, we don't have to re-execute [`crate_def_map_query`]. In fact, we only
need to re-execute it when we add/remove new files or when we change mod
declarations.

[`block_def_map_query`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/nameres.rs#L326-L354

We store the resulting modules in a `Vec`-based indexed arena. The indices in
the arena becomes module IDs. And this brings us to the next topic:
assigning IDs in the general case.

## Location Interner pattern

One way to assign IDs is how we've dealt with modules: Collect all items into a
single array in some specific order and use the index in the array as an ID. The
main drawback of this approach is that these IDs are not stable: Adding a new item can
shift the IDs of all other items. This works for modules, because adding a module is
a comparatively rare operation, but would be less convenient for, for example,
functions.

Another solution here is positional IDs: We can identify a function as "the
function with name `foo` in a ModuleId(92) module". Such locations are stable:
adding a new function to the module (unless it is also named `foo`) does not
change the location. However, such "ID" types ceases to be a `Copy`able integer and in
general can become pretty large if we account for nesting (for example: "third parameter of
the `foo` function of the `bar` `impl` in the `baz` module").

[`Intern` and `Lookup`] traits allows us to combine the benefits of positional and numeric
IDs. Implementing both traits effectively creates a bidirectional append-only map
between locations and integer IDs (typically newtype wrappers for [`salsa::InternId`])
which can "intern" a location and return an integer ID back. The salsa database we use
includes a couple of [interners]. How to "garbage collect" unused locations
is an open question.

[`Intern` and `Lookup`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-expand/src/lib.rs#L96-L106
[interners]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-expand/src/lib.rs#L108-L122
[`salsa::InternId`]: https://docs.rs/salsa/0.16.1/salsa/struct.InternId.html

For example, we use `Intern` and `Lookup` implementations to assign IDs to
definitions of functions, structs, enums, etc. The location, [`ItemLoc`] contains
two bits of information:

* the ID of the module which contains the definition,
* the ID of the specific item in the module's source code.

We "could" use a text offset for the location of a particular item, but that would play
badly with salsa: offsets change after edits. So, as a rule of thumb, we avoid
using offsets, text ranges or syntax trees as keys and values for queries. What
we do instead is we store "index" of the item among all of the items of a file
(so, a positional based ID, but localized to a single file).

[`ItemLoc`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/lib.rs#L209-L212

One thing we've glossed over for the time being is support for macros. We have
only proof of concept handling of macros at the moment, but they are extremely
interesting from an "assigning IDs" perspective.

## Macros and recursive locations

The tricky bit about macros is that they effectively create new source files.
While we can use `FileId`s to refer to original files, we can't just assign them
willy-nilly to the pseudo files of macro expansion. Instead, we use a special
ID, [`HirFileId`] to refer to either a usual file or a macro-generated file:

```rust
enum HirFileId {
    FileId(FileId),
    Macro(MacroCallId),
}
```

`MacroCallId` is an interned ID that identifies a particular macro invocation.
Simplifying, it's a `HirFileId` of a file containing the call plus the offset
of the macro call in the file.

Note how `HirFileId` is defined in terms of `MacroCallId` which is defined in
terms of `HirFileId`! This does not recur infinitely though: any chain of
`HirFileId`s bottoms out in `HirFileId::FileId`, that is, some source file
actually written by the user.

Note also that in the actual implementation, the two variants are encoded in
a single `u32`, which are differentiated by the MSB (most significant bit).
If the MSB is 0, the value represents a `FileId`, otherwise the remaining
31 bits represent a `MacroCallId`.

[`HirFileId`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/span/src/lib.rs#L148-L160

Now that we understand how to identify a definition, in a source or in a
macro-generated file, we can discuss name resolution a bit.

## Name resolution

Name resolution faces the same problem as the module tree: if we look at the
syntax tree directly, we'll have to recompute name resolution after every
modification. The solution to the problem is the same: We [lower] the source code of
each module into a position-independent representation which does not change if
we modify bodies of the items. After that we [loop] resolving all imports until
we've reached a fixed point.

[lower]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/item_tree.rs#L110-L154
[loop]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/nameres/collector.rs#L404-L437
And, given all our preparation with IDs and a position-independent representation,
it is satisfying to [test] that typing inside function body does not invalidate
name resolution results.

[test]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/nameres/tests/incremental.rs#L31

An interesting fact about name resolution is that it "erases" all of the
intermediate paths from the imports: in the end, we know which items are defined
and which items are imported in each module, but, if the import was `use
foo::bar::baz`, we deliberately forget what modules `foo` and `bar` resolve to.

To serve "goto definition" requests on intermediate segments we need this info
in the IDE, however. Luckily, we need it only for a tiny fraction of imports, so we just ask
the module explicitly, "What does the path `foo::bar` resolve to?". This is a
general pattern: we try to compute the minimal possible amount of information
during analysis while allowing IDE to ask for additional specific bits.

Name resolution is also a good place to introduce another salsa pattern used
throughout the analyzer:

## Source Map pattern

Due to an obscure edge case in completion, IDE needs to know the syntax node of
a use statement which imported the given completion candidate. We can't just
store the syntax node as a part of name resolution: this will break
incrementality, due to the fact that syntax changes after every file
modification.

We solve this problem during the lowering step of name resolution. Along with
the [`ItemTree`] output, the lowering query additionally produces an [`AstIdMap`]
via an [`ast_id_map`] query. The `ItemTree` contains [imports], but in a
position-independent form based on [`AstId`]. The `AstIdMap` contains a mapping
from position-independent `AstId`s to (position-dependent) syntax nodes.

[`ItemTree`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/item_tree.rs
[`AstIdMap`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-expand/src/ast_id_map.rs#L136-L142
[`ast_id_map`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/item_tree/lower.rs#L32
[imports]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/item_tree.rs#L559-L563
[`AstId`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-expand/src/ast_id_map.rs#L29


## Type inference

First of all, implementation of type inference in rust-analyzer was spearheaded
by [@flodiebold]. [#327] was an awesome Christmas present, thank you, Florian!

Type inference runs on per-function granularity and uses the patterns we've
discussed previously.

First, we [lower the AST] of a function body into a position-independent
representation. In this representation, each expression is assigned a
[positional ID]. Alongside the lowered expression, [a source map] is produced,
which maps between expression ids and original syntax. This lowering step also
deals with "incomplete" source trees by replacing missing expressions by an
explicit `Missing` expression.

Given the lowered body of the function, we can now run [type inference] and
construct a mapping from `ExprId`s to types.

[@flodiebold]: https://github.com/flodiebold
[#327]: https://github.com/rust-lang/rust-analyzer/pull/327
[lower the AST]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/body.rs
[positional ID]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/hir.rs#L37
[a source map]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-def/src/body.rs#L84-L88
[type inference]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/hir-ty/src/infer.rs#L76-L131

## Tying it all together: completion

To conclude the overview of the rust-analyzer, let's trace the request for
(type-inference powered!) code completion!

We start by [receiving a message] from the language client. We decode the
message as a request for completion and [schedule it on the threadpool]. This is
the place where we [catch] canceled errors if, immediately after completion, the
client sends some modification.

In [the handler], we deserialize LSP requests into rust-analyzer specific data
types (by converting a file url into a numeric `FileId`), [ask analysis for
completion] and serialize results into the LSP.

The [completion implementation] is finally the place where we start doing the actual
work. The first step is to collect the [`CompletionContext`] -- a struct which
describes the cursor position in terms of Rust syntax and semantics. For
example, `expected_name: Option<NameOrNameRef>` is the syntactic representation
for the expected name of what we're completing (usually the parameter name of
a function argument), while `expected_type: Option<Type>` is the semantic model
for the expected type of what we're completing.

To construct the context, we first do an ["IntelliJ Trick"]: we insert a dummy
identifier at the cursor's position and parse this modified file, to get a
reasonably looking syntax tree. Then we do a bunch of "classification" routines
to figure out the context. For example, we [find an parent `fn` node], get a
[semantic model] for it (using the lossy `source_analyzer` infrastructure)
and use it to determine the [expected type at the cursor position].

The second step is to run a [series of independent completion routines]. Let's
take a closer look at [`complete_dot`], which completes fields and methods in
`foo.bar|`. First we extract a semantic receiver type out of the `DotAccess`
argument. Then, using the semantic model for the type, we determine if the
receiver implements the `Future` trait, and add a `.await` completion item in
the affirmative case. Finally, we add all fields & methods from the type to
completion.

[receiving a message]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/main_loop.rs#L213
[schedule it on the threadpool]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/dispatch.rs#L197-L211
[catch]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/dispatch.rs#L292
[the handler]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/rust-analyzer/src/handlers/request.rs#L850-L876
[ask analysis for completion]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide/src/lib.rs#L605-L615
[completion implementation]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/lib.rs#L148-L229
[`CompletionContext`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/context.rs#L407-L441
["IntelliJ Trick"]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/context.rs#L644-L648
[find an parent `fn` node]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/context/analysis.rs#L463
[semantic model]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/context/analysis.rs#L466
[expected type at the cursor position]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/context/analysis.rs#L467
[series of independent completion routines]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/lib.rs#L157-L226
[`complete_dot`]: https://github.com/rust-lang/rust-analyzer/blob/2024-01-01/crates/ide-completion/src/completions/dot.rs#L11-L41
