// compile-flags: -Ztrait-solver=next

fn main() {
    let x;
    x = Box::new(x);
    //~^ ERROR mismatched types
    //~| NOTE cyclic type of infinite size
}
