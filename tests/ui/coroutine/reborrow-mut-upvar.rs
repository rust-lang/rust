// build-pass

#![feature(coroutines)]

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
