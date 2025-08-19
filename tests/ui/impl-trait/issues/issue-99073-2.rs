use std::fmt::Display;

fn main() {
    test("hi", true);
}

fn test<T: Display>(t: T, recurse: bool) -> impl Display {
    let f = || {
        let i: u32 = test::<i32>(-1, false);
        //~^ ERROR expected generic type parameter, found `i32`
        println!("{i}");
    };
    if recurse {
        f();
    }
    t
}
