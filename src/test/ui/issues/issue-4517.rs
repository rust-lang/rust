fn bar(int_param: usize) {}

fn main() {
    let foo: [u8; 4] = [1; 4];
    bar(foo);
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `[u8; 4]`
    //~| expected usize, found array of 4 elements
}
