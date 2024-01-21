extern "C" {
    fn foo(x: usize, ...);
}

fn test() -> u8 {
    127
}

fn main() {
    foo(1, test);
    //~^ ERROR can't pass `{fn item test: fn() -> u8}` to variadic function
}
