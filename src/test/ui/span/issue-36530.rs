// gate-test-custom_inner_attributes

#[foo] //~ ERROR is currently unknown to the compiler
mod foo {
    #![foo] //~ ERROR is currently unknown to the compiler
            //~| ERROR non-builtin inner attributes are unstable
}

fn main() {}
