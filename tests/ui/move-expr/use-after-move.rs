#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x = vec![1, 2, 3];
    let _c = || {
    //~^ ERROR borrow of moved value: `x`
        let y = move(x);
        println!("{x:?}");
        drop(y);
    };
}
