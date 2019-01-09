// normalize-stderr-test: "existed:.*\(" -> "existed: $$FILE_NOT_FOUND_MSG ("

// test that errors in a (selection) of macros don't kill compilation
// immediately, so that we get more errors listed at a time.

#![feature(asm)]
#![feature(trace_macros, concat_idents)]

#[derive(Default)] //~ ERROR
enum OrDeriveThis {}

fn main() {
    asm!(invalid); //~ ERROR

    concat_idents!("not", "idents"); //~ ERROR

    option_env!(invalid); //~ ERROR
    env!(invalid); //~ ERROR
    env!(foo, abr, baz); //~ ERROR
    env!("RUST_HOPEFULLY_THIS_DOESNT_EXIST"); //~ ERROR

    format!(invalid); //~ ERROR

    include!(invalid); //~ ERROR

    include_str!(invalid); //~ ERROR
    include_str!("i'd be quite surprised if a file with this name existed"); //~ ERROR
    include_bytes!(invalid); //~ ERROR
    include_bytes!("i'd be quite surprised if a file with this name existed"); //~ ERROR

    trace_macros!(invalid); //~ ERROR
}
