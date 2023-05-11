// run-pass

#![feature(generators)]

fn _run(bar: &mut i32) {
    || { //~ WARN unused generator that must be used
        {
            let _baz = &*bar;
            yield;
        }

        *bar = 2;
    };
}

fn main() {}
