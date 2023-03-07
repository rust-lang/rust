use std::mem;

fn main() {
    mem::transmute(0);
    //~^ ERROR type annotations needed [E0282]
}
