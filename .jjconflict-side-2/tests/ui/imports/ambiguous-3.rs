// https://github.com/rust-lang/rust/issues/47525

fn main() {
    use a::*;
    x();
    //~^ ERROR `x` is ambiguous
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
