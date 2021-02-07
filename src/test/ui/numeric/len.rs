fn main() {
    let array = [1, 2, 3];
    test(array.len()); //~ ERROR arguments to this function are incorrect
}

fn test(length: u32) {
    println!("{}", length);
}
