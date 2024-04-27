unsafe fn f() { return; }

fn main() {
    f();
    //~^ ERROR E0133
}
