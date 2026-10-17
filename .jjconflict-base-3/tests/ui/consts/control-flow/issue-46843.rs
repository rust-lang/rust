enum Thing {
    This,
    That,
}

fn non_const() -> Thing {
    Thing::This
}

pub const Q: i32 = match non_const() {
    //~^ ERROR cannot call non-const function
    Thing::This => 1,
    Thing::That => 0
};

fn main() {}
