fn foo(_x: u32) {
    _x = 4;
    //~^ ERROR cannot assign to immutable argument `_x`
}

fn main() {}
