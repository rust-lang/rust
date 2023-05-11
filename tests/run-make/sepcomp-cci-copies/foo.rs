extern crate cci_lib;
use cci_lib::cci_fn;

fn call1() -> usize {
    cci_fn()
}

mod a {
    use cci_lib::cci_fn;
    pub fn call2() -> usize {
        cci_fn()
    }
}

mod b {
    pub fn call3() -> usize {
        0
    }
}

fn main() {
    call1();
    a::call2();
    b::call3();
}
