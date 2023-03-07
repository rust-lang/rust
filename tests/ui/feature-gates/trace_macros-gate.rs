// Test that the trace_macros feature gate is on.

fn main() {
    trace_macros!(); //~ ERROR `trace_macros` is not stable
                     //~| ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(true); //~ ERROR `trace_macros` is not stable
    trace_macros!(false); //~ ERROR `trace_macros` is not stable

    macro_rules! expando {
        ($x: ident) => { trace_macros!($x) } //~ ERROR `trace_macros` is not stable
    }

    expando!(true);
}
