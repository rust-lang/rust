// run-pass

#![feature(generators)]

fn _run(bar: &mut i32) {
    || {
        {
            let _baz = &*bar;
            yield;
        }

        *bar = 2;
    };
}

fn main() {}
