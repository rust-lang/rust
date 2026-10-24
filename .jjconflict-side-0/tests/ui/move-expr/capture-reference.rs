#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let c = {
        let x = 22;
        || {
            let y = move(&x);
            //~^ ERROR `x` does not live long enough
            println!("{y:?}");
        }
    };

    c();
}
