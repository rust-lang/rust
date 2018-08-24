fn main() {
    let array = [std::env::args().len()];
    array[1]; //~ ERROR index out of bounds
}
