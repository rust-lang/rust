fn foo() -> isize {
    let x: isize;

    while 1 != 2  {
        break;
        x = 0;
    }

    println!("{}", x); //~ ERROR borrow of possibly uninitialized variable: `x`

    return 17;
}

fn main() { println!("{}", foo()); }
