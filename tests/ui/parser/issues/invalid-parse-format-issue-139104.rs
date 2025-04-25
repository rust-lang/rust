fn main() {
    println!("{foo:_1.4}", foo = 3.14); //~ ERROR invalid format string: tuple index access isn't supported
    println!("{foo:1.4_1.4}", foo = 3.14); //~ ERROR invalid format string: tuple index access isn't supported
    println!("xxx{0:_1.4", 1.11); //~ ERROR invalid format string: expected `}`, found `.`
    println!("{foo:_1.4", foo = 3.14); //~ ERROR invalid format string: expected `}`, found `.`
    println!("xxx{0:_1.4", 1.11); //~ ERROR invalid format string: expected `}`, found `.`
    println!("xxx{  0", 1.11); //~ ERROR invalid format string: expected `}`, found `0`
}
