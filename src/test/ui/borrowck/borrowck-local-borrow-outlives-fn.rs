fn cplusplus_mode(x: isize) -> &'static isize {
    &x
    //~^ ERROR cannot return reference to function parameter `x` [E0515]
}

fn main() {}
