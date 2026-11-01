//@ run-pass

#![feature(coroutines)]

fn _run(bar: &mut i32) {
    #[coroutine] || { //~ WARN unused coroutine that must be used
        {
            let _baz = &*bar;
            yield;
        }

        *bar = 2;
    };
}

fn main() {}
