// gate-test-custom_inner_attributes

#![feature(custom_attribute)]

#[foo]
mod foo {
    #![foo] //~ ERROR non-builtin inner attributes are unstable
}

fn main() {}
