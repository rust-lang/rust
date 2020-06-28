enum Thing {
    This,
    That,
}

fn non_const() -> Thing {
    Thing::This
}

pub const Q: i32 = match non_const() {
    //~^ ERROR calls in constants are limited to constant functions
    Thing::This => 1,
    Thing::That => 0
};

fn main() {}
