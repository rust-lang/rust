macro_rules! f {
    ($abi:literal) => {
        extern $abi fn f() {}
    }
}

f!("Foo"__);
//~^ ERROR suffixes on string literals are invalid

fn main() {}
