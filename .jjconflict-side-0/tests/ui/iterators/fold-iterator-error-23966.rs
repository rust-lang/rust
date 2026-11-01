// https://github.com/rust-lang/rust/issues/23966
fn main() {
    "".chars().fold(|_, _| (), ());
    //~^ ERROR E0277
}
