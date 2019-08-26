fn main() {
    let caller<F> = |f: F|  //~ ERROR expected one of `:`, `;`, `=`, `@`, or `|`, found `<`
    where F: Fn() -> i32
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
