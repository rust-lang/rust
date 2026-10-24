#![feature(coroutines)]

fn main() {
    let x = (|_| {},);

    #[coroutine] || {
        let x = x;

        x.0({ //~ ERROR borrow may still be in use when coroutine yields
            yield;
        });
    };
}
