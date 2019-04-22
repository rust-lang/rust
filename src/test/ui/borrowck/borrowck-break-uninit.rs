fn foo() -> isize {
    let x: isize;

    loop {
        break;
        x = 0;
    }

    println!("{}", x); //~ ERROR borrow of possibly uninitialized variable: `x`

    return 17;
}

fn main() { println!("{}", foo()); }
