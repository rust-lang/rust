macro_rules! f {
    ($abi:literal) => {
        extern $abi fn f() {} //~ WARN missing_abi
    }
}

f!("Foo"__);
//~^ ERROR suffixes on string literals are invalid

fn main() {}
