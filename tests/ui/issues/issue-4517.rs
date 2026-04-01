//@ dont-require-annotations: NOTE

fn bar(int_param: usize) {}

fn main() {
    let foo: [u8; 4] = [1; 4];
    bar(foo);
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `[u8; 4]`
}
