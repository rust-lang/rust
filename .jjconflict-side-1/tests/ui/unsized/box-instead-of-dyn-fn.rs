use std::fmt::Debug;

// Test to suggest boxing the return type, and the closure branch of the `if`

fn print_on_or_the_other<'a>(a: i32, b: &'a String) -> dyn Fn() + 'a {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    if a % 2 == 0 {
        move || println!("{a}")
    } else {
        Box::new(move || println!("{}", b))
    }
}

fn main() {}
