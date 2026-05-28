#![feature(rustc_attrs)]

macro_rules! check {
    ($expr: expr) => (
        #[rustc_dummy = $expr]
        use main as _;
    );
}

check!("0"); // OK
check!(0); // OK
check!(0u8); //~ ERROR suffixed literals are not allowed in attributes
check!(-0); //~ ERROR attribute value must be a literal
check!(0 + 0); //~ ERROR attribute value must be a literal

fn main() {}
