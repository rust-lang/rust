fn main() {
    let array = [1, 2, 3];
    test(array.len()); //~ ERROR mismatched types
}

fn test(length: u32) {
    println!("{}", length);
}
