#![feature(trace_macros)]

fn main() {
    trace_macros!(); //~ ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(1); //~ ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(ident); //~ ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(for); //~ ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(true,); //~ ERROR trace_macros! accepts only `true` or `false`
    trace_macros!(false 1); //~ ERROR trace_macros! accepts only `true` or `false`


    // should be fine:
    macro_rules! expando {
        ($x: ident) => { trace_macros!($x) }
    }

    expando!(true);
}
