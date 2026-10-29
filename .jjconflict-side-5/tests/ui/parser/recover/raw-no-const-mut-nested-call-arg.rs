// Regression test for https://github.com/rust-lang/rust/issues/157853.

fn main() {
    Test([&raw 2])
    //~^ ERROR expected one of
}
