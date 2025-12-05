//@ run-pass

#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let a = #[coroutine]
    || {
        {
            let w: i32 = 4;
            yield;
            println!("{:?}", w);
        }
        {
            let x: i32 = 5;
            yield;
            println!("{:?}", x);
        }
        {
            let y: i32 = 6;
            yield;
            println!("{:?}", y);
        }
        {
            let z: i32 = 7;
            yield;
            println!("{:?}", z);
        }
    };
    assert_eq!(8, std::mem::size_of_val(&a));
}
