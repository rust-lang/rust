mod to_reuse {
    pub fn foo() {}
}

reuse to_reuse::foo;
//~^ ERROR functions delegation is not yet fully implemented

fn main() {}
