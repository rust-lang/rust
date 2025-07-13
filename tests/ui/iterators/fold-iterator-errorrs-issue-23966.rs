fn main() {
    "".chars().fold(|_, _| (), ());
    //~^ ERROR E0277
}

// https://github.com/rust-lang/rust/issues/23966
