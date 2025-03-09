macro_rules! f {
    ($abi:literal) => {
        extern $abi fn f() {} //~ WARN missing_abi
        //~^ WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust future!
    }
}

f!("Foo"__);
//~^ ERROR suffixes on string literals are invalid

fn main() {}
