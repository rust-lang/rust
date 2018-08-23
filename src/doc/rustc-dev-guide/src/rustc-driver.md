# The Rustc Driver

The [`rustc_driver`] is essentially `rustc`'s `main()` function. It acts as
the glue for running the various phases of the compiler in the correct order,
managing state such as the [`SourceMap`] \(maps AST nodes to source code),
[`Session`] \(general build context and error messaging) and the [`TyCtxt`]
\(the "typing context", allowing you to query the type system and other cool
stuff). The `rustc_driver` crate also provides external users with a method
for running code at particular times during the compilation process, allowing
third parties to effectively use `rustc`'s internals as a library for
analysing a crate or emulating the compiler in-process (e.g. the RLS).

For those using `rustc` as a library, the `run_compiler()` function is the main
entrypoint to the compiler. Its main parameters are a list of command-line
arguments and a reference to something which implements the `CompilerCalls`
trait. A `CompilerCalls` creates the overall `CompileController`, letting it
govern which compiler passes are run and attach callbacks to be fired at the end
of each phase.

From `rustc_driver`'s perspective, the main phases of the compiler are:

1. *Parse Input:* Initial crate parsing
2. *Configure and Expand:* Resolve `#[cfg]` attributes, name resolution, and
   expand macros
3. *Run Analysis Passes:* Run trait resolution, typechecking, region checking
   and other miscellaneous analysis passes on the crate
4. *Translate to LLVM:* Translate to the in-memory form of LLVM IR and turn it
   into an executable/object files

The `CompileController` then gives users the ability to inspect the ongoing
compilation process

- after parsing
- after AST expansion
- after HIR lowering
- after analysis, and
- when compilation is done

The `CompileState`'s various `state_after_*()` constructors can be inspected to
determine what bits of information are available to which callback.

For a more detailed explanation on using `rustc_driver`, check out the
[stupid-stats] guide by `@nrc` (attached as [Appendix A]).

> **Warning:** By its very nature, the internal compiler APIs are always going
> to be unstable. That said, we do try not to break things unnecessarily.

## A Note On Lifetimes

The Rust compiler is a fairly large program containing lots of big data
structures (e.g. the AST, HIR, and the type system) and as such, arenas and
references are heavily relied upon to minimize unnecessary memory use. This
manifests itself in the way people can plug into the compiler, preferring a
"push"-style API (callbacks) instead of the more Rust-ic "pull" style (think
the `Iterator` trait).

For example the [`CompileState`], the state passed to callbacks after each
phase, is essentially just a box of optional references to pieces inside the
compiler. The lifetime bound on the `CompilerCalls` trait then helps to ensure
compiler internals don't "escape" the compiler (e.g. if you tried to keep a
reference to the AST after the compiler is finished), while still letting users
record *some* state for use after the `run_compiler()` function finishes.

Thread-local storage and interning are used a lot through the compiler to reduce
duplication while also preventing a lot of the ergonomic issues due to many
pervasive lifetimes. The `rustc::ty::tls` module is used to access these
thread-locals, although you should rarely need to touch it.


[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/
[`CompileState`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/driver/struct.CompileState.html
[`Session`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyCtxt.html
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceMap.html
[stupid-stats]: https://github.com/nrc/stupid-stats
[Appendix A]: appendix/stupid-stats.html
