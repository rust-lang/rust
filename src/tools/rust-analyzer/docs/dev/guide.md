# Guide to rust-analyzer

## About the guide

This guide describes the current state of rust-analyzer as of 2019-01-20 (git
tag [guide-2019-01]). Its purpose is to document various problems and
architectural solutions related to the problem of building IDE-first compiler
for Rust. There is a video version of this guide as well:
https://youtu.be/ANKBNiSWyfc.

[guide-2019-01]: https://github.com/rust-lang/rust-analyzer/tree/guide-2019-01

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

[`AnalysisHost`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L265-L284
[`Analysis`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L291-L478

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
`Change`. [`Change`] is a builder for a single change
"transaction", so it suffices to study its methods to understand all of the
input data.

[`Change`]: https://github.com/rust-lang/rust-analyzer/blob/master/crates/base_db/src/change.rs#L14-L89

The `(add|change|remove)_file` methods control the set of the input files, where
each file has an integer id (`FileId`, picked by the client), text (`String`)
and a filesystem path. Paths are tricky; they'll be explained below, in source roots
section, together with the `add_root` method. The `add_library` method allows us to add a
group of files which are assumed to rarely change. It's mostly an optimization
and does not change the fundamental picture.

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
directory on a file systems, like `/home/matklad/projects/rustraytracer/**.rs`.

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

Now let's see how the `Analysis` API is exposed via the JSON RPC based language server protocol. The
hard part here is managing changes (which can come either from the file system
or from the editor) and concurrency (we want to spawn background jobs for things
like syntax highlighting). We use the event loop pattern to manage the zoo, and
the loop is the [`main_loop_inner`] function. The [`main_loop`] does a one-time
initialization and tearing down of the resources.

[`main_loop`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L51-L110
[`main_loop_inner`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L156-L258


Let's walk through a typical analyzer session!

First, we need to figure out what to analyze. To do this, we run `cargo
metadata` to learn about Cargo packages for current workspace and dependencies,
and we run `rustc --print sysroot` and scan the "sysroot" (the directory containing the current Rust toolchain's files) to learn about crates like
`std`. Currently we load this configuration once at the start of the server, but
it should be possible to dynamically reconfigure it later without restart.

[main_loop.rs#L62-L70](https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L62-L70)

The [`ProjectModel`] we get after this step is very Cargo and sysroot specific,
it needs to be lowered to get the input in the form of `Change`. This
happens in [`ServerWorldState::new`] method. Specifically

* Create a `SourceRoot` for each Cargo package and sysroot.
* Schedule a filesystem scan of the roots.
* Create an analyzer's `Crate` for each Cargo **target** and sysroot crate.
* Setup dependencies between the crates.

[`ProjectModel`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/project_model.rs#L16-L20
[`ServerWorldState::new`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L38-L160

The results of the scan (which may take a while) will be processed in the body
of the main loop, just like any other change. Here's where we handle:

* [File system changes](https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L194)
* [Changes from the editor](https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L377)

After a single loop's turn, we group the changes into one `Change` and
[apply] it. This always happens on the main thread and blocks the loop.

[apply]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L216

To handle requests, like ["goto definition"], we create an instance of the
`Analysis` and [`schedule`] the task (which consumes `Analysis`) on the
threadpool. [The task] calls the corresponding `Analysis` method, while
massaging the types into the LSP representation. Keep in mind that if we are
executing "goto definition" on the threadpool and a new change comes in, the
task will be canceled as soon as the main loop calls `apply_change` on the
`AnalysisHost`.

["goto definition"]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L216
[`schedule`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L426-L455
[The task]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop/handlers.rs#L205-L223

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
`AnalysisHost` types are newtype wrappers for [`RootDatabase`] -- a salsa
database.

[`RootDatabase`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ide_api/src/db.rs#L88-L134

Salsa input queries are defined in [`FilesDatabase`] (which is a part of
`RootDatabase`). They closely mirror the familiar `Change` structure:
indeed, what `apply_change` does is it sets the values of input queries.

[`FilesDatabase`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/base_db/src/input.rs#L150-L174

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
this reason and gets handled by the [`source_binder`].

[`source_binder`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/source_binder.rs

The semantic interface is declared in the [`code_model_api`] module. Each entity is
identified by an integer ID and has a bunch of methods which take a salsa database
as an argument and returns other entities (which are also IDs). Internally, these
methods invoke various queries on the database to build the model on demand.
Here's [the list of queries].

[`code_model_api`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/code_model_api.rs
[the list of queries]: https://github.com/rust-lang/rust-analyzer/blob/7e84440e25e19529e4ff8a66e521d1b06349c6ec/crates/ra_hir/src/db.rs#L20-L106

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
[rust-specific]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_syntax/src/ast/generated.rs

The next step in constructing the semantic model is ...

## Building a Module Tree

The algorithm for building a tree of modules is to start with a crate root
(remember, each `Crate` from a `CrateGraph` has a `FileId`), collect all `mod`
declarations and recursively process child modules. This is handled by the
[`module_tree_query`], with two slight variations.

[`module_tree_query`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/module_tree.rs#L115-L133

First, rust-analyzer builds a module tree for all crates in a source root
simultaneously. The main reason for this is historical (`module_tree` predates
`CrateGraph`), but this approach also enables accounting for files which are not
part of any crate. That is, if you create a file but do not include it as a
submodule anywhere, you still get semantic completion, and you get a warning
about a free-floating module (the actual warning is not implemented yet).

The second difference is that `module_tree_query` does not *directly* depend on
the "parse" query (which is confusingly called `source_file`). Why would calling
the parse directly be bad? Suppose the user changes the file slightly, by adding
an insignificant whitespace. Adding whitespace changes the parse tree (because
it includes whitespace), and that means recomputing the whole module tree.

We deal with this problem by introducing an intermediate [`submodules_query`].
This query processes the syntax tree and extracts a set of declared submodule
names. Now, changing the whitespace results in `submodules_query` being
re-executed for a *single* module, but because the result of this query stays
the same, we don't have to re-execute [`module_tree_query`]. In fact, we only
need to re-execute it when we add/remove new files or when we change mod
declarations.

[`submodules_query`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/module_tree.rs#L41

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

[`LocationInterner`] allows us to combine the benefits of positional and numeric
IDs. It is a bidirectional append-only map between locations and consecutive
integers which can "intern" a location and return an integer ID back. The salsa
database we use includes a couple of [interners]. How to "garbage collect"
unused locations is an open question.

[`LocationInterner`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_db/src/loc2id.rs#L65-L71
[interners]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/db.rs#L22-L23

For example, we use `LocationInterner` to assign IDs to definitions of functions,
structs, enums, etc. The location, [`DefLoc`] contains two bits of information:

* the ID of the module which contains the definition,
* the ID of the specific item in the modules source code.

We "could" use a text offset for the location of a particular item, but that would play
badly with salsa: offsets change after edits. So, as a rule of thumb, we avoid
using offsets, text ranges or syntax trees as keys and values for queries. What
we do instead is we store "index" of the item among all of the items of a file
(so, a positional based ID, but localized to a single file).

[`DefLoc`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/ids.rs#L129-L139

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

`MacroCallId` is an interned ID that specifies a particular macro invocation.
Its `MacroCallLoc` contains:

* `ModuleId` of the containing module
* `HirFileId` of the containing file or pseudo file
* an index of this particular macro invocation in this file (positional id
  again).

Note how `HirFileId` is defined in terms of `MacroCallLoc` which is defined in
terms of `HirFileId`! This does not recur infinitely though: any chain of
`HirFileId`s bottoms out in `HirFileId::FileId`, that is, some source file
actually written by the user.

[`HirFileId`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/ids.rs#L31-L93

Now that we understand how to identify a definition, in a source or in a
macro-generated file, we can discuss name resolution a bit.

## Name resolution

Name resolution faces the same problem as the module tree: if we look at the
syntax tree directly, we'll have to recompute name resolution after every
modification. The solution to the problem is the same: We [lower] the source code of
each module into a position-independent representation which does not change if
we modify bodies of the items. After that we [loop] resolving all imports until
we've reached a fixed point.

[lower]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres/lower.rs#L113-L147
[loop]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres.rs#L186-L196
And, given all our preparation with IDs and a position-independent representation,
it is satisfying to [test] that typing inside function body does not invalidate
name resolution results.

[test]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres/tests.rs#L376

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

We solve this problem during the lowering step of name resolution. The lowering
query actually produces a *pair* of outputs: `LoweredModule` and [`SourceMap`].
The `LoweredModule` module contains [imports], but in a position-independent form.
The `SourceMap` contains a mapping from position-independent imports to
(position-dependent) syntax nodes.

The result of this basic lowering query changes after every modification. But
there's an intermediate [projection query] which returns only the first
position-independent part of the lowering. The result of this query is stable.
Naturally, name resolution [uses] this stable projection query.

[imports]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres/lower.rs#L52-L59
[`SourceMap`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres/lower.rs#L52-L59
[projection query]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/nameres/lower.rs#L97-L103
[uses]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/query_definitions.rs#L49

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
[lower the AST]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/expr.rs
[positional ID]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/expr.rs#L13-L15
[a source map]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/expr.rs#L41-L44
[type inference]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/ty.rs#L1208-L1223

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
work. The first step is to collect the `CompletionContext` -- a struct which
describes the cursor position in terms of Rust syntax and semantics. For
example, `function_syntax: Option<&'a ast::FnDef>` stores a reference to
the enclosing function *syntax*, while `function: Option<hir::Function>` is the
`Def` for this function.

To construct the context, we first do an ["IntelliJ Trick"]: we insert a dummy
identifier at the cursor's position and parse this modified file, to get a
reasonably looking syntax tree. Then we do a bunch of "classification" routines
to figure out the context. For example, we [find an ancestor `fn` node] and we get a
[semantic model] for it (using the lossy `source_binder` infrastructure).

The second step is to run a [series of independent completion routines]. Let's
take a closer look at [`complete_dot`], which completes fields and methods in
`foo.bar|`. First we extract a semantic function and a syntactic receiver
expression out of the `Context`. Then we run type-inference for this single
function and map our syntactic expression to `ExprId`. Using the ID, we figure
out the type of the receiver expression. Then we add all fields & methods from
the type to completion.

[receiving a message]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L203
[schedule it on the threadpool]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L428
[catch]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L436-L442
[the handler]: https://salsa.zulipchat.com/#narrow/stream/181542-rfcs.2Fsalsa-query-group/topic/design.20next.20steps
[ask analysis for completion]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ide_api/src/lib.rs#L439-L444
[ask analysis for completion]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L439-L444
[completion implementation]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion.rs#L46-L62
[`CompletionContext`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion/completion_context.rs#L14-L37
["IntelliJ Trick"]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion/completion_context.rs#L72-L75
[find an ancestor `fn` node]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion/completion_context.rs#L116-L120
[semantic model]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion/completion_context.rs#L123
[series of independent completion routines]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion.rs#L52-L59
[`complete_dot`]: https://github.com/rust-lang/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/completion/complete_dot.rs#L6-L22
