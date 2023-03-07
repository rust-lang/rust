// run-pass
// pretty-expanded FIXME #23616

#![feature(trace_macros, log_syntax)]

// make sure these macros can be used as in the various places that
// macros can occur.

// items
trace_macros!(false);
log_syntax!();

fn main() {

    // statements
    trace_macros!(false);
    log_syntax!();

    // expressions
    (trace_macros!(false),
     log_syntax!());
}
