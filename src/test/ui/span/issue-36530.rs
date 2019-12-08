// gate-test-custom_inner_attributes

#![feature(register_attr)]

#![register_attr(foo)]

#[foo]
mod foo {
    #![foo] //~ ERROR non-builtin inner attributes are unstable
}

fn main() {}
