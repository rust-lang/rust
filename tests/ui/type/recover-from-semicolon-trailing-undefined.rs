//@ compile-flags: -Znext-solver=globally

// Regression test for https://github.com/rust-lang/rust/issues/151610

fn main() {
    let x_str = {
        x!("{}", x);
        //~^ ERROR cannot find macro `x` in this scope
    };
    println!("{}", x_str);
    //~^ ERROR `()` doesn't implement `std::fmt::Display`
}
