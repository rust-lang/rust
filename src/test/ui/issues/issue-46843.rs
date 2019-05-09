enum Thing { This, That }

fn non_const() -> Thing {
    Thing::This
}

pub const Q: i32 = match non_const() {
    //~^ ERROR E0015
    //~^^ ERROR unimplemented expression type
    Thing::This => 1, //~ ERROR unimplemented expression type
    Thing::That => 0
};

fn main() {}
