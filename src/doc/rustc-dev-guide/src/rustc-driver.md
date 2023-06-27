# `rustc_driver` and `rustc_interface`

The [`rustc_driver`] is essentially `rustc`'s `main()` function. It acts as
the glue for running the various phases of the compiler in the correct order,
using the interface defined in the [`rustc_interface`] crate.

The `rustc_interface` crate provides external users with an (unstable) API
for running code at particular times during the compilation process, allowing
third parties to effectively use `rustc`'s internals as a library for
analyzing a crate or emulating the compiler in-process (e.g. rustdoc).

For those using `rustc` as a library, the [`rustc_interface::run_compiler()`][i_rc]
function is the main entrypoint to the compiler. It takes a configuration for the compiler
and a closure that takes a [`Compiler`]. `run_compiler` creates a `Compiler` from the
configuration and passes it to the closure. Inside the closure, you can use the `Compiler`
to drive queries to compile a crate and get the results. This is what the `rustc_driver` does too.
You can see a minimal example of how to use `rustc_interface` [here][example].

You can see what queries are currently available through the rustdocs for [`Compiler`].
You can see an example of how to use them by looking at the `rustc_driver` implementation,
specifically the [`rustc_driver::run_compiler` function][rd_rc] (not to be confused with
[`rustc_interface::run_compiler`][i_rc]). The `rustc_driver::run_compiler` function
takes a bunch of command-line args and some other configurations and
drives the compilation to completion.

`rustc_driver::run_compiler` also takes a [`Callbacks`][cb],
a trait that allows for custom compiler configuration,
as well as allowing some custom code run after different phases of the compilation.

> **Warning:** By its very nature, the internal compiler APIs are always going
> to be unstable. That said, we do try not to break things unnecessarily.


[cb]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/trait.Callbacks.html
[rd_rc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver_impl/fn.run_compiler.html
[i_rc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/fn.run_compiler.html
[example]: https://github.com/rust-lang/rustc-dev-guide/blob/master/examples/rustc-driver-example.rs
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/
[`Compiler`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Compiler.html
[`Session`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html
[stupid-stats]: https://github.com/nrc/stupid-stats
[Appendix A]: appendix/stupid-stats.html
