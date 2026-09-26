fn test2() -> impl Fn() -> (str, u8) {
    || loop {}
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}

fn main() {}
