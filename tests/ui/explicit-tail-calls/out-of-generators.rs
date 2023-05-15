#![feature(explicit_tail_calls)]
#![feature(generators)]

fn main() {
    let _generator = || {
        yield 1;
        become f() //~ error: mismatched function ABIs
    };
}

fn f() {}
