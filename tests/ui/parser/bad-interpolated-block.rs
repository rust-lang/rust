fn main() {}

macro_rules! m {
    ($b:block) => {
        'lab: $b; //~ ERROR cannot use a `block` macro fragment here
        unsafe $b; //~ ERROR cannot use a `block` macro fragment here
        |x: u8| -> () $b; //~ ERROR cannot use a `block` macro fragment here
    }
}

fn foo() {
    m!({});
}
