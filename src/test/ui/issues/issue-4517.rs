fn bar(int_param: usize) {}

fn main() {
    let foo: [u8; 4] = [1; 4];
    bar(foo);
    //~^ ERROR arguments to this function are incorrect
    //~| expected `usize`, found array `[u8; 4]`
}
