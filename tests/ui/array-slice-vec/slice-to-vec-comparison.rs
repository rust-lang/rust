fn main() {
    let a = &[];
    let b: &Vec<u8> = &vec![];
    a > b;
    //~^ ERROR mismatched types
}
