#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x = vec![1, 2, 3];
    let _c = || {
        let y = move(x);
        let z = move(x);
        //~^ ERROR use of moved value: `x`
        drop(y);
        drop(z);
    };
}
