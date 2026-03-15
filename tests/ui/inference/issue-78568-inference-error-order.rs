//! Regression test for issue #78568
//!
//! This test ensures that type inference errors are reported with the correct
//! error ordering. Previously, the compiler would emit a confusing error about
//! `Display` not being implemented for `()` in the `println!` macro, instead
//! of pointing to the actual type inference problem at the `parse()` call.

fn main() {
    let guess = "123";
    let guess = match guess.trim().parse() {
        //~^ ERROR type annotations needed
        Ok(num) => num + 1,
        Err(_) => todo!(),
    };
    println!("guess: {}", guess);
}