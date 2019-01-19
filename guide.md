# Guide to rust-analyzer

## About the guide

This guide describes the current start of the rust-analyzer as of 2019-01-20
(commit hash guide-2019-01). Its purpose is to
document various problems and architectural solutions related to the problem of
building IDE-first compiler.

## The big picture

On the highest possible level, rust analyzer is a stateful component. Client may
apply changes to the analyzer (new contents of `foo.rs` file is "fn main() {}")
and it may ask semantic questions about the current state (what is the
definition of the identifier with offset 92 in file `bar.rs`?). Two important
properties hold:

* Analyzer does not do any IO. It starts in an empty state and all input data is
  provided via `apply_change` API.

* Only queries about the current state are supported. One can, of course,
  simulate undo and redo by keeping log of changes and inverse-changes.

## IDE API

To see this big picture, let's take a look at the [`AnalysisHost`] and
[`Analysis`] pair of types. `AnalysisHost` has three methods:

* `default` for creating an empty analysis
* `apply_change(&mut self)` to make changes (this is how you get from an empty
  state to something interesting)
* `analysis(&self)` to get an instance of `Analysis`

`Analysis` has a ton of methods for IDEs, like `goto_definition`, or
`completions`. Both inputs and outputs of `Analysis`' methods are formulated in
terms of files and offsets, and **not** in terms of Rust concepts like structs,
traits, etc. The "typed" API with Rust specific types is slightly lower in the
stack, we'll talk about it later.

[`AnalysisHost`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L265-L284
[`Analysis`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L291-L478

The reason for `Analysis` and `AnalysisHost` separation is that we want apply
changes "uniquely", but we might want to fork an `Analysis` and send it to
another thread for background processing. That is, there is only a single
`AnalysisHost`, but there may be several (equivalent) `Analysis`.

Note that all of the `Analysis` API return `Cancelable<T>`. This is required to
be responsive in IDE setting. Sometimes a long-running query is being computed
and the user types something in the editor and asks for completion. In this
case, we cancel the long-running computation (so it returns `Err(Canceled)`),
apply the change and execute request for completion. We never use stale data to
answer requests. Under the cover, `AnalysisHost` "remembers" all outstanding
`Analysis` instances. `AnalysisHost::apply_change` method cancels all
`Analysis`es, blocks until of them are `Dropped` and then applies change
in-place. This is the familiar to rustaceans read-write lock interior
mutability.

Next, lets talk about what are inputs to the Analysis, precisely.

## Inputs

Rust Analyzer never does any IO itself, all inputs get passed explicitly via
`AnalysisHost::apply_change` method, which accepts a single argument:
`AnalysisChange`. [`AnalysisChange`] is a builder for a single change
"transaction", so it suffices to study its methods to understand all of the
input data.

[`AnalysisChange`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/lib.rs#L119-L167

The `(add|change|remove)_file` methods control the set of the input files, where
each file has an integer id (`FileId`, picked by the client), text (`String`)
and a filesystem path. Paths are tricky, they'll be explained in source roots
section, together with `add_root` method. `add_library` method allows to add a
group of files which are assumed to rarely change. It's mostly an optimization
and does not change fundamental picture.

`set_crate_graph` method allows to control how the input files are partitioned
into compilation unites -- crates. It also controls (in theory, not implemented
yet) `cfg` flags. `CrateGraph` is a directed acyclic graph of crates. Each crate
has a root `FileId`, a set of active `cfg` flags and a set of dependencies. Each
dependency is a pair of a crate and a name. It is possible to have two crates
with the same root `FileId` but different `cfg`-flags/dependencies. This model
is lower than Cargo's model of packages: each Cargo package consists of several
targets, each of which is a separate crate (or several crates, if you try
different feature combinations).

Procedural macros should become inputs as well, but currently they are not
supported. Procedural macro will be a black box `Box<dyn Fn(TokenStream) -> TokenStream>`
function, and will be inserted into the crate graph just like dependencies.

Soon we'll talk how we build an LSP server on top of `Analysis`, but first,
let's deal with that paths issue.


## Source roots (aka filesystems are horrible)

This is a non-essential section, feel free to skip.

The previous section said that the file system path is an attribute of a file,
but this is not a whole truth. Making it an absolute `PathBuf` will be bad for
several reasons. First, file-systems are full of (platform-dependent) edge cases:

* it's hard (requires a syscall) to decide if two paths are equivalent
* some file-systems are case-sensitive
* paths are not necessary UTF-8
* symlinks can form cycles

Second, this might hurt reproducibility and hermeticity of builds. In theory,
moving a project from `/foo/bar/my-project` to `/spam/eggs/my-project` should
not change a bit in the output. However, if absolute path is a part of the
input, it is at least in theory observable, and *could* affect the output.

Yet another problem is that we really-really want to avoid doing IO, but with
Rust the set of "input" files is not necessary known up-front. In theory, you
can have `#[path="/dev/random"] mod foo;`.

To solve (or explicitly refuse to solve) these problems rust analyzer uses the
concept of source root. Roughly speaking, source roots is a contents of a
directory on a file systems, like `/home/matklad/projects/rustraytracer/**.rs`.

More precisely, all files (`FileId`s) are partitioned into disjoint
`SourceRoot`s. Each file has a relative utf-8 path within the `SourceRoot`.
`SourceRoot` has an identity (integer id). Crucially, the root path of the
source root itself is unknown to the analyzer: client is supposed to maintain a
mapping between SourceRoot ids (which are assigned by the client) and actual
`PathBuf`s. `SourceRoot`s give a sane tree model of the file system to the
analyzer.

Note that `mod`, `#[path]` and `include!()` can only reference files from the
same source root. It is of course is possible to explicitly add extra files to
the source root, even `/dev/random`.

## Language Server Protocol

Now let's see how `Analysis` API is exposed via JSON RPC based LSP protocol. The
hard part here is managing changes (which can come either from the file system
or from the editor) and concurrency (we want to spawn background jobs for things
like syntax highlighting). We use the event loop pattern to manage the zoo, and
the loop is the [`main_loop_inner`] function. The [`main_loop`] does a one-time
initialization and tearing down of the resources.

[`main_loop`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L51-L110
[`main_loop_inner`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L156-L258


Let's walk through a typical analyzer session!

First, we need to figure out what to analyze. To do this, we run `cargo
metadata` to learn about Cargo packages for current workspace and dependencies,
and we run `rustc --print sysroot` and scan sysroot to learn about crates like
`std`. Currently we load this configuration once at the start of the server, but
it should be possible to dynamically reconfigure it later without restart.

[main_loop.rs#L62-L70](https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L62-L70)

The [`ProjectModel`] we get after this step is very Cargo and sysroot specific,
it needs to be lowered to get the input in the form of `AnalysisChange`. This
happens in [`ServerWorldState::new`] method. Specifically

* Create a `SourceRoot` for each Cargo package and sysroot.
* Schedule a file system scan of the roots.
* Create an analyzer's `Crate` for each Cargo **target** and sysroot crate.
* Setup dependencies between the crates.

[`ProjectModel`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/project_model.rs#L16-L20
[`ServerWorldState::new`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L38-L160

The results of the scan (which may take a while) will be processed in the body
of the main loop, just like any other change. Here's where we handle

* [File system changes](https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L194)
* [Changes from the editor](https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L377)

After a single loop's turn, we group them into one `AnalysisChange` and
[apply] it. This always happens on the main thread and blocks the loop.

[apply]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L216

To handle requests, like ["goto definition"], we create an instance of the
`Analysis` and [`schedule`] the task (which consumes `Analysis`) onto
threadpool. [The task] calls the corresponding `Analysis` method, while
massaging the types into the LSP representation. Keep in mind that if we are
executing "goto definition" on the threadpool and a new change comes in, the
task will be canceled as soon as the main loop calls `apply_change` on the
`AnalysisHost`.

["goto definition"]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/server_world.rs#L216
[`schedule`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop.rs#L426-L455
[The task]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_lsp_server/src/main_loop/handlers.rs#L205-L223

This concludes the overview of the analyzer's programing *interface*. Next, lets
dig into the implementation!

## Salsa

The most straightforward way to implement "apply change, get analysis, repeat"
API would be to maintain the input state and to compute all possible analysis
information from scratch after every change. This works, but scales poorly with
the size of the project. To make this fast, we need to take advantage of the
fact that most of the changes are small, and that analysis results are unlikely
to change significantly between invocations.

To do this we use [salsa]: a framework for incremental on-demand computation.
You can skip the rest of the section if you are familiar with rustc red-green
algorithm.

[salsa]: https://github.com/salsa-rs/salsa

It's better to refer to salsa's docs to learn about it. Here's a small excerpt:

The key idea of salsa is that you define your program as a set of queries. Every
query is used like function K -> V that maps from some key of type K to a value
of type V. Queries come in two basic varieties:

* **Inputs**: the base inputs to your system. You can change these whenever you
  like.

* **Functions**: pure functions (no side effects) that transform your inputs
  into other values. The results of queries is memoized to avoid recomputing
  them a lot. When you make changes to the inputs, we'll figure out (fairly
  intelligently) when we can re-use these memoized values and when we have to
  recompute them.


For further discussion, its important to understand one bit of "fairly
intelligently". Suppose we have to functions, `f1` and `f2`, and one input, `i`.
We call `f1(X)` which in turn calls `f2(Y)` which inspects `i(Z)`. `i(Z)`
returns some value `V1`, `f2` uses that and returns `R1`, `f1` uses that and
returns `O`. Now, let's change `i` at `Z` to `V2` from `V1` and try to compute
`f1(X)` again. Because `f1(X)` (transitively) depends on `i(Z)`, we can't just
reuse its value as is. However, if `f2(Y)` is *still* equal to `R1` (despite the
`i`'s change), we, in fact, *can* reuse `O` as result of `f1(X)`. And that's how
salsa works: it recomputes results in *reverse* order, starting from inputs and
progressing towards outputs, stopping as soon as it sees an intermediate value
that hasn't changed.

## Salsa Input Queries

All analyzer information is stored in a salsa database. `Analysis` and
`AnalysisHost` types are newtype wrappers for [`RootDatabase`] -- a salsa
database.

[`RootDatabase`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_ide_api/src/db.rs#L88-L134

Salsa input queries are defined in [`FilesDatabase`] (which is a part of
`RootDatabase`). They closely mirror the familiar `AnalysisChange` structure:
indeed, what `apply_change` does is it sets the values of input queries.

[`FilesDatabase`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_db/src/input.rs#L150-L174

## From text to semantic model

The bulk of the rust-analyzer is transforming input text into semantic model of
Rust code: a web of entities like modules, structs, functions and traits.

An important fact to realize is that (unlike most other languages like C# or
Java) there isn't a one-to-one mapping between source code and semantic model. A
single function definition in the source code might result in several semantic
functions: for example, the same source file might be included as a module into
several crate, or a single "crate" might be present in the compilation DAG
several times, with different sets of `cfg`s enabled.

The semantic interface is declared in [`code_model_api`] module. Each entity is
identified by integer id and has a bunch of methods which take a salsa database
as an argument and returns other entities (which are ids). Internally, this
methods invoke various queries on the database to build the model on demand.
Here's [the list of queries].

[`code_model_api`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/code_model_api.rs
[the list of queries]: https://github.com/rust-analyzer/rust-analyzer/blob/7e84440e25e19529e4ff8a66e521d1b06349c6ec/crates/ra_hir/src/db.rs#L20-L106

The first step of building the model is parsing the source code.

## Syntax trees

An important property of the Rust language is that each file can be parsed in
isolation. Unlike, say, `C++`, an `include` can't change the meaning of the
syntax. For this reason, Rust analyzer can build a syntax tree for each "source
file", which could then be reused by several semantic models if this file
happens to be a part of several crates.

Rust analyzer uses a similar representation of syntax trees to that of `Roslyn`
and Swift's new
[libsyntax](https://github.com/apple/swift/tree/5e2c815edfd758f9b1309ce07bfc01c4bc20ec23/lib/Syntax).
Swift's docs give an excellent overview of the approach, so I skip this part
here and instead outline the main characteristics of the syntax trees:

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

* Syntax trees do not know the file they are build from, they only know about
  the text.

The implementation is based on the generic [rowan] crate on top of which a
[rust-specific] AST is generated.

[rowan]: https://github.com/rust-analyzer/rowan/tree/100a36dc820eb393b74abe0d20ddf99077b61f88
[rust-specific]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_syntax/src/ast/generated.rs

The next step in constructing the semantic model is ...

## Building a Module Tree

The algorithm for building a tree of modules is to start with a crate root
(remember, each `Crate` from a `CrateGraph` has a `FileId`), collect all mod
declarations and recursively process child modules. This is handled by the
[`module_tree_query`](https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/module_tree.rs#L116-L123),
with a two slight variations.

First, rust analyzer builds a module tree for all crates in a source root
simultaneously. The main reason for this is historical (`module_tree` predates
`CrateGraph`), but this approach also allows to account for files which are not
part of any crate. That is, if you create a file but do not include it as a
submodule anywhere, you still get semantic completion, and you get a warning
about free-floating module (the actual warning is not implemented yet).

The second difference is that `module_tree_query` does not *directly* depend on
the "parse" query (which is confusingly called `source_file`). Why would calling
the parse directly be bad? Suppose the user changes the file slightly, by adding
an insignificant whitespace. Adding whitespace changes the parse tree (because
it includes whitespace), and that means recomputing the whole module tree.

We deal with this problem by introducing an intermediate [`submodules_query`].
This query processes the syntax tree an extract a set of declared submodule
names. Now, changing the whitespace results in `submodules_query` being
re-executed for a *single* module, but because the result of this query stays
the same, we don't have to re-execute [`module_tree_query`]. In fact, we only
need to re-execute it when we add/remove new files or when we change mod
declarations,

[`submodules_query`]: https://github.com/rust-analyzer/rust-analyzer/blob/guide-2019-01/crates/ra_hir/src/module_tree.rs#L41)





## Location Interner pattern

## Macros and recursive locations

## Name resolution

## Source Map pattern

## Tying it all together: completion
