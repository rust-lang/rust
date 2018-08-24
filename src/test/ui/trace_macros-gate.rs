// Test that the trace_macros feature gate is on.

fn main() {
    trace_macros!(); //~ ERROR `trace_macros` is not stable
    trace_macros!(1); //~ ERROR `trace_macros` is not stable
    trace_macros!(ident); //~ ERROR `trace_macros` is not stable
    trace_macros!(for); //~ ERROR `trace_macros` is not stable
    trace_macros!(true,); //~ ERROR `trace_macros` is not stable
    trace_macros!(false 1); //~ ERROR `trace_macros` is not stable

    // Errors are signalled early for the above, before expansion.
    // See trace_macros-gate2 and trace_macros-gate3. for examples
    // of the below being caught.

    macro_rules! expando {
        ($x: ident) => { trace_macros!($x) } //~ ERROR `trace_macros` is not stable
    }

    expando!(true);
}
