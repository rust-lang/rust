struct A(u8);

fn main() {
    let a: A;
    a.0 = 1; //~ ERROR partially assigned binding `a` isn't fully initialized
    //~^ ERROR E0658
}
