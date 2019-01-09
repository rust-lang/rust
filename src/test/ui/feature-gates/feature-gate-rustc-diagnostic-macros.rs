// Test that diagnostic macros are gated by `rustc_diagnostic_macros` feature
// gate

__register_diagnostic!(E0001);
//~^ ERROR cannot find macro `__register_diagnostic!` in this scope

fn main() {
    __diagnostic_used!(E0001);
    //~^ ERROR cannot find macro `__diagnostic_used!` in this scope
}

__build_diagnostic_array!(DIAGNOSTICS);
//~^ ERROR cannot find macro `__build_diagnostic_array!` in this scope
