# `rustc_driver` and `rustc_interface`

The [`rustc_driver`] is essentially `rustc`'s `main` function.
It acts as the glue for running the various phases of the compiler in the correct order,
using the interface defined in the [`rustc_interface`] crate.

Generally the [`rustc_interface`] crate provides external users with an (unstable) API
for running code at particular times during the compilation process, allowing
third parties to effectively use `rustc`'s internals as a library for
analyzing a crate or for ad hoc emulating of the compiler (i.e. `rustdoc`
compiling code and serving output).

More specifically the [`rustc_interface::run_compiler`][i_rc] function is the
main entrypoint for using [`nightly-rustc`] as a library.
Initially [`run_compiler`][i_rc] takes a configuration variable for the compiler
and a `closure` taking a yet unresolved [`Compiler`].
Operationally [`run_compiler`][i_rc] creates a `Compiler` from the configuration and passes
it to the `closure`. Inside the `closure` you can use the `Compiler` to drive
queries to compile a crate and get the results.
Providing results about the internal state of the compiler what the [`rustc_driver`] does too.
You can see a minimal example of how to use [`rustc_interface`] [here][example].

You can see what queries are currently available in the [`Compiler`] rustdocs.
You can see an example of how to use the queries by looking at the `rustc_driver` implementation,
specifically [`rustc_driver::run_compiler`][rd_rc]
(not to be confused with [`rustc_interface::run_compiler`][i_rc]).
Generally [`rustc_driver::run_compiler`][i_rc] takes a bunch of command-line args
and some other configurations and drives the compilation to completion.

Finally [`rustc_driver::run_compiler`][rd_rc] also takes a [`Callbacks`][cb],
which is a `trait` that allows for custom compiler configuration,
as well as allowing custom code to run after different phases of the compilation.

> **Warning:** By its very nature, the internal compiler APIs are always going
> to be unstable. That said, we do try not to break things unnecessarily.


[`Compiler`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Compiler.html
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`Session`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html
[Appendix A]: appendix/stupid-stats.html
[cb]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/trait.Callbacks.html
[example]: https://github.com/rust-lang/rustc-dev-guide/blob/master/examples/rustc-driver-example.rs
[i_rc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/fn.run_compiler.html
[rd_rc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver_impl/fn.run_compiler.html
[stupid-stats]: https://github.com/nrc/stupid-stats
[`nightly-rustc`]: https://doc.rust-lang.org/nightly/nightly-rustc/
