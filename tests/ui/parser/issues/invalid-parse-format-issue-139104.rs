fn main() {
    println!("{foo:_1.4}", foo = 3.14); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{0:_1.4}", 1.11); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{:_1.4}", 3.14); //~ ERROR invalid format string: expected `}`, found `.`

    println!("{foo:_1.4", foo = 3.14); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{0:_1.4", 1.11); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{:_1.4", 3.14); //~ ERROR invalid format string: expected `}`, found `.`

    println!("{  0", 1.11); //~ ERROR invalid format string: expected `}`, found `0`
    println!("{foo:1.4_1.4}", foo = 3.14); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{0:1.4_1.4}", 3.14); //~ ERROR invalid format string: expected `}`, found `.`
}
