//@compile-flags: -Z track-diagnostics
//@no-rustfix

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@normalize-stderr-test: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@normalize-stderr-test: "src/tools/clippy/" -> ""

#![warn(clippy::let_and_return, clippy::unnecessary_cast)]

fn main() {
    // Check the provenance of a lint sent through `LintContext::span_lint()`
    let a = 3u32;
    let b = a as u32;
    //~^ unnecessary_cast

    // Check the provenance of a lint sent through `TyCtxt::node_span_lint()`
    let c = {
        let d = 42;
        d
        //~^ let_and_return
    };
}
