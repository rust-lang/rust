# The Rustc Driver and Interface

The [`rustc_driver`] is essentially `rustc`'s `main()` function. It acts as
the glue for running the various phases of the compiler in the correct order,
using the interface defined in the [`rustc_interface`] crate.

The `rustc_interface` crate provides external users with an (unstable) API
for running code at particular times during the compilation process, allowing
third parties to effectively use `rustc`'s internals as a library for
analysing a crate or emulating the compiler in-process (e.g. the RLS or rustdoc).

For those using `rustc` as a library, the `interface::run_compiler()` function is the main
entrypoint to the compiler. It takes a configuration for the compiler and a closure that
takes a [`Compiler`]. `run_compiler` creates a `Compiler` from the configuration and passes
it to the closure. Inside the closure, you can use the `Compiler` to drive queries to compile
a crate and get the results. This is what the `rustc_driver` does too.

You can see what queries are currently available through the rustdocs for [`Compiler`].
You can see an example of how to use them by looking at the `rustc_driver` implementation,
specifically the [`rustc_driver::run_compiler` function][rd_rc] (not to be confused with
`interface::run_compiler`). The `rustc_driver::run_compiler` function takes a bunch of
command-line args and some other configurations and drives the compilation to completion.

> **Warning:** By its very nature, the internal compiler APIs are always going
> to be unstable. That said, we do try not to break things unnecessarily.

## A Note On Lifetimes

The Rust compiler is a fairly large program containing lots of big data
structures (e.g. the AST, HIR, and the type system) and as such, arenas and
references are heavily relied upon to minimize unnecessary memory use. This
manifests itself in the way people can plug into the compiler, preferring a
"push"-style API (callbacks) instead of the more Rust-ic "pull" style (think
the `Iterator` trait).

Thread-local storage and interning are used a lot through the compiler to reduce
duplication while also preventing a lot of the ergonomic issues due to many
pervasive lifetimes. The `rustc::ty::tls` module is used to access these
thread-locals, although you should rarely need to touch it.

[rd_rc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/fn.run_compiler.html
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/
[`Compiler`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Compiler.html
[`Session`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyCtxt.html
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceMap.html
[stupid-stats]: https://github.com/nrc/stupid-stats
[Appendix A]: appendix/stupid-stats.html
