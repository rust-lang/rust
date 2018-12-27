#![feature(generators)]

fn main() {
    let x = (|_| {},);

    || {
        let x = x;

        x.0({ //~ ERROR borrow may still be in use when generator yields
            yield;
        });
    };
}
