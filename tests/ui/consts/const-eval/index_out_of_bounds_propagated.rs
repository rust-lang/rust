//@ build-fail

fn main() {
    let array = [std::env::args().len()];
    array[1]; //~ ERROR operation will panic
}
