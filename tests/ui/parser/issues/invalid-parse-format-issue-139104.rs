fn main() {
    println!("{foo:_1.4}", foo = 3.14); //~ ERROR invalid format string: invalid format string for argument `foo`
    println!("{foo:1.4_1.4}", foo = 3.14); //~ ERROR invalid format string: invalid format string for argument `foo`
    println!("xxx{0:_1.4}", 1.11); //~ ERROR invalid format string: invalid format string for argument at index `0`
    println!("{foo:_1.4", foo = 3.14); //~ ERROR invalid format string: invalid format string for argument `foo`
    println!("xxx{0:_1.4", 1.11); //~ ERROR invalid format string: invalid format string for argument at index `0`
    println!("xxx{  0", 1.11); //~ ERROR invalid format string: expected `}`, found `0`
}
