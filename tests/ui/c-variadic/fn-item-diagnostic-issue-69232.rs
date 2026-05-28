// https://github.com/rust-lang/rust/issues/69232

extern "C" {
    fn foo(x: usize, ...);
}

fn test() -> u8 {
    127
}

fn main() {
    unsafe { foo(1, test) }; //~ ERROR can't pass a function item to a variadic function
}
