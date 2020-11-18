// gate-test-custom_inner_attributes

#![feature(register_attr)]

#![register_attr(foo)]

#[foo]
mod foo {
    #![foo] //~ ERROR custom inner attributes are unstable
}

fn main() {}
