fn f(a: u16, b: &str) {}

fn f2(a: u16) {}

fn main() {
    f(0);
    //~^ ERROR E0061
    //~| expected 2 parameters

    f2();
    //~^ ERROR E0061
    //~| expected 1 parameter
}
