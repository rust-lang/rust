//@ run-rustfix

#[allow(dead_code)]
enum Demo {
    A = 1,
    B == 2 //~ ERROR unexpected `==`
    //~^ ERROR expected item, found `==`
}

fn main() {}
