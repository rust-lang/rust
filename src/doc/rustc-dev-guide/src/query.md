# Queries: demand-driven compilation

As described in [the high-level overview of the compiler][hl], the
Rust compiler is current transitioning from a traditional "pass-based"
setup to a "demand-driven" system. **The Compiler Query System is the
key to our new demand-driven organization.** The idea is pretty
simple. You have various queries that compute things about the input
-- for example, there is a query called `type_of(def_id)` that, given
the def-id of some item, will compute the type of that item and return
it to you.

[hl]: high-level-overview.html

Query execution is **memoized** – so the first time you invoke a
query, it will go do the computation, but the next time, the result is
returned from a hashtable. Moreover, query execution fits nicely into
**incremental computation**; the idea is roughly that, when you do a
query, the result **may** be returned to you by loading stored data
from disk (but that's a separate topic we won't discuss further here).

The overall vision is that, eventually, the entire compiler
control-flow will be query driven. There will effectively be one
top-level query ("compile") that will run compilation on a crate; this
will in turn demand information about that crate, starting from the
*end*.  For example:

- This "compile" query might demand to get a list of codegen-units
  (i.e. modules that need to be compiled by LLVM).
- But computing the list of codegen-units would invoke some subquery
  that returns the list of all modules defined in the Rust source.
- That query in turn would invoke something asking for the HIR.
- This keeps going further and further back until we wind up doing the
  actual parsing.

However, that vision is not fully realized. Still, big chunks of the
compiler (for example, generating MIR) work exactly like this.

### Invoking queries

To invoke a query is simple. The tcx ("type context") offers a method
for each defined query. So, for example, to invoke the `type_of`
query, you would just do this:

```rust,ignore
let ty = tcx.type_of(some_def_id);
```

### Cycles between queries

A cycle is when a query becomes stuck in a loop e.g. query A generates query B
which generates query A again.

Currently, cycles during query execution should always result in a
compilation error. Typically, they arise because of illegal programs
that contain cyclic references they shouldn't (though sometimes they
arise because of compiler bugs, in which case we need to factor our
queries in a more fine-grained fashion to avoid them).

However, it is nonetheless often useful to *recover* from a cycle
(after reporting an error, say) and try to soldier on, so as to give a
better user experience. In order to recover from a cycle, you don't
get to use the nice method-call-style syntax. Instead, you invoke
using the `try_get` method, which looks roughly like this:

```rust,ignore
use ty::queries;
...
match queries::type_of::try_get(tcx, DUMMY_SP, self.did) {
  Ok(result) => {
    // no cycle occurred! You can use `result`
  }
  Err(err) => {
    // A cycle occurred! The error value `err` is a `DiagnosticBuilder`,
    // meaning essentially an "in-progress", not-yet-reported error message.
    // See below for more details on what to do here.
  }
}
```

So, if you get back an `Err` from `try_get`, then a cycle *did* occur. This
means that you must ensure that a compiler error message is reported. You can
do that in two ways:

The simplest is to invoke `err.emit()`. This will emit the cycle error to the
user.

However, often cycles happen because of an illegal program, and you
know at that point that an error either already has been reported or
will be reported due to this cycle by some other bit of code. In that
case, you can invoke `err.cancel()` to not emit any error. It is
traditional to then invoke:

```rust,ignore
tcx.sess.delay_span_bug(some_span, "some message")
```

`delay_span_bug()` is a helper that says: we expect a compilation
error to have happened or to happen in the future; so, if compilation
ultimately succeeds, make an ICE with the message `"some
message"`. This is basically just a precaution in case you are wrong.

### How the compiler executes a query

So you may be wondering what happens when you invoke a query
method. The answer is that, for each query, the compiler maintains a
cache – if your query has already been executed, then, the answer is
simple: we clone the return value out of the cache and return it
(therefore, you should try to ensure that the return types of queries
are cheaply cloneable; insert a `Rc` if necessary).

#### Providers

If, however, the query is *not* in the cache, then the compiler will
try to find a suitable **provider**. A provider is a function that has
been defined and linked into the compiler somewhere that contains the
code to compute the result of the query.

**Providers are defined per-crate.** The compiler maintains,
internally, a table of providers for every crate, at least
conceptually. Right now, there are really two sets: the providers for
queries about the **local crate** (that is, the one being compiled)
and providers for queries about **external crates** (that is,
dependencies of the local crate). Note that what determines the crate
that a query is targeting is not the *kind* of query, but the *key*.
For example, when you invoke `tcx.type_of(def_id)`, that could be a
local query or an external query, depending on what crate the `def_id`
is referring to (see the `self::keys::Key` trait for more information
on how that works).

Providers always have the same signature:

```rust,ignore
fn provider<'cx, 'tcx>(tcx: TyCtxt<'cx, 'tcx, 'tcx>,
                       key: QUERY_KEY)
                       -> QUERY_RESULT
{
    ...
}
```

Providers take two arguments: the `tcx` and the query key. Note also
that they take the *global* tcx (i.e. they use the `'tcx` lifetime
twice), rather than taking a tcx with some active inference context.
They return the result of the query.

####  How providers are setup

When the tcx is created, it is given the providers by its creator using
the `Providers` struct. This struct is generated by the macros here, but it
is basically a big list of function pointers:

```rust,ignore
struct Providers {
    type_of: for<'cx, 'tcx> fn(TyCtxt<'cx, 'tcx, 'tcx>, DefId) -> Ty<'tcx>,
    ...
}
```

At present, we have one copy of the struct for local crates, and one
for external crates, though the plan is that we may eventually have
one per crate.

These `Provider` structs are ultimately created and populated by
`librustc_driver`, but it does this by distributing the work
throughout the other `rustc_*` crates. This is done by invoking
various `provide` functions. These functions tend to look something
like this:

```rust,ignore
pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        type_of,
        ..*providers
    };
}
```

That is, they take an `&mut Providers` and mutate it in place. Usually
we use the formulation above just because it looks nice, but you could
as well do `providers.type_of = type_of`, which would be equivalent.
(Here, `type_of` would be a top-level function, defined as we saw
before.) So, if we want to add a provider for some other query,
let's call it `fubar`, into the crate above, we might modify the `provide()`
function like so:

```rust,ignore
pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        type_of,
        fubar,
        ..*providers
    };
}

fn fubar<'cx, 'tcx>(tcx: TyCtxt<'cx, 'tcx>, key: DefId) -> Fubar<'tcx> { ... }
```

N.B. Most of the `rustc_*` crates only provide **local
providers**. Almost all **extern providers** wind up going through the
[`rustc_metadata` crate][rustc_metadata], which loads the information from the
crate metadata. But in some cases there are crates that provide queries for
*both* local and external crates, in which case they define both a
`provide` and a `provide_extern` function that `rustc_driver` can
invoke.

[rustc_metadata]: https://github.com/rust-lang/rust/tree/master/src/librustc_metadata

### Adding a new kind of query

So suppose you want to add a new kind of query, how do you do so?
Well, defining a query takes place in two steps:

1. first, you have to specify the query name and arguments; and then,
2. you have to supply query providers where needed.

To specify the query name and arguments, you simply add an entry to
the big macro invocation in
[`src/librustc/ty/query/mod.rs`][query-mod], which looks something like:

[query-mod]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/query/index.html

```rust,ignore
define_queries! { <'tcx>
    /// Records the type of every item.
    [] fn type_of: TypeOfItem(DefId) -> Ty<'tcx>,

    ...
}
```

Each line of the macro defines one query. The name is broken up like this:

```rust,ignore
[] fn type_of: TypeOfItem(DefId) -> Ty<'tcx>,
^^    ^^^^^^^  ^^^^^^^^^^ ^^^^^     ^^^^^^^^
|     |        |          |         |
|     |        |          |         result type of query
|     |        |          query key type
|     |        dep-node constructor
|     name of query
query flags
```

Let's go over them one by one:

- **Query flags:** these are largely unused right now, but the intention
  is that we'll be able to customize various aspects of how the query is
  processed.
- **Name of query:** the name of the query method
  (`tcx.type_of(..)`). Also used as the name of a struct
  (`ty::queries::type_of`) that will be generated to represent
  this query.
- **Dep-node constructor:** indicates the constructor function that
  connects this query to incremental compilation. Typically, this is a
  `DepNode` variant, which can be added by modifying the
  `define_dep_nodes!` macro invocation in
  [`librustc/dep_graph/dep_node.rs`][dep-node].
  - However, sometimes we use a custom function, in which case the
    name will be in snake case and the function will be defined at the
    bottom of the file. This is typically used when the query key is
    not a def-id, or just not the type that the dep-node expects.
- **Query key type:** the type of the argument to this query.
  This type must implement the `ty::query::keys::Key` trait, which
  defines (for example) how to map it to a crate, and so forth.
- **Result type of query:** the type produced by this query. This type
  should (a) not use `RefCell` or other interior mutability and (b) be
  cheaply cloneable. Interning or using `Rc` or `Arc` is recommended for
  non-trivial data types.
  - The one exception to those rules is the `ty::steal::Steal` type,
    which is used to cheaply modify MIR in place. See the definition
    of `Steal` for more details. New uses of `Steal` should **not** be
    added without alerting `@rust-lang/compiler`.

[dep-node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/dep_graph/struct.DepNode.html

So, to add a query:

- Add an entry to `define_queries!` using the format above.
- Possibly add a corresponding entry to the dep-node macro.
- Link the provider by modifying the appropriate `provide` method;
  or add a new one if needed and ensure that `rustc_driver` is invoking it.

#### Query structs and descriptions

For each kind, the `define_queries` macro will generate a "query struct"
named after the query. This struct is a kind of a place-holder
describing the query. Each such struct implements the
`self::config::QueryConfig` trait, which has associated types for the
key/value of that particular query. Basically the code generated looks something
like this:

```rust,ignore
// Dummy struct representing a particular kind of query:
pub struct type_of<'tcx> { phantom: PhantomData<&'tcx ()> }

impl<'tcx> QueryConfig for type_of<'tcx> {
  type Key = DefId;
  type Value = Ty<'tcx>;
}
```

There is an additional trait that you may wish to implement called
`self::config::QueryDescription`. This trait is used during cycle
errors to give a "human readable" name for the query, so that we can
summarize what was happening when the cycle occurred. Implementing
this trait is optional if the query key is `DefId`, but if you *don't*
implement it, you get a pretty generic error ("processing `foo`...").
You can put new impls into the `config` module. They look something like this:

```rust,ignore
impl<'tcx> QueryDescription for queries::type_of<'tcx> {
    fn describe(tcx: TyCtxt, key: DefId) -> String {
        format!("computing the type of `{}`", tcx.item_path_str(key))
    }
}
```

