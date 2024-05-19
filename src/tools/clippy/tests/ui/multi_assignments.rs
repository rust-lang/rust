#![warn(clippy::multi_assignments)]
fn main() {
    let (mut a, mut b, mut c, mut d) = ((), (), (), ());
    a = b = c;
    //~^ ERROR: assignments don't nest intuitively
    //~| NOTE: `-D clippy::multi-assignments` implied by `-D warnings`
    a = b = c = d;
    //~^ ERROR: assignments don't nest intuitively
    //~| ERROR: assignments don't nest intuitively
    a = b = { c };
    //~^ ERROR: assignments don't nest intuitively
    a = { b = c };
    //~^ ERROR: assignments don't nest intuitively
    a = (b = c);
    //~^ ERROR: assignments don't nest intuitively
}
