fn main() {
    let caller<F> = |f: F|
    where F: Fn() -> i32 //~ ERROR expected expression, found keyword `where`
    {
        let x = f();
        println!("Y {}",x);
        return x;
    };

    caller(bar_handler);
}

fn bar_handler() -> i32 {
    5
}
