// https://github.com/rust-lang/rust/issues/47525

fn main() {
    use a::*;
    x();
    //~^ ERROR `x` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

mod a {
    mod b {
        pub fn x() { println!(module_path!()); }
    }
    mod c {
        pub fn x() { println!(module_path!()); }
    }

    pub use self::b::*;
    pub use self::c::*;
}
