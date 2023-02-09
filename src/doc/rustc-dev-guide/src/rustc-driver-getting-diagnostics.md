# Example: Getting diagnostic through `rustc_interface`

`rustc_interface` allows you to intercept diagnostics that would otherwise be printed to stderr.

## Getting diagnostics

To get diagnostics from the compiler,
configure `rustc_interface::Config` to output diagnostic to a buffer,
and run `TyCtxt.analysis`. The following was tested
with <!-- date-check: Feb 2023 --> `nightly-2023-02-06` (See [here][example]
for the complete example):

[example]: https://github.com/rust-lang/rustc-dev-guide/blob/master/examples/rustc-driver-getting-diagnostics.rs

```rust
let buffer = sync::Arc::new(sync::Mutex::new(Vec::new()));
let config = rustc_interface::Config {
    opts: config::Options {
        // Configure the compiler to emit diagnostics in compact JSON format.
        error_format: config::ErrorOutputType::Json {
            pretty: false,
            json_rendered: rustc_errors::emitter::HumanReadableErrorType::Default(
                rustc_errors::emitter::ColorConfig::Never,
            ),
        },
        /* other config */
    },   
    /* other config */
};
rustc_interface::run_compiler(config, |compiler| {
    compiler.enter(|queries| {
        queries.global_ctxt().unwrap().enter(|tcx| {
            // Run the analysis phase on the local crate to trigger the type error.
            let _ = tcx.analysis(());
        });
    });
});
// Read buffered diagnostics.
let diagnostics = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
```
