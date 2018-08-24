#[derive(Debug)]
enum Numbers {
    Three
}

pub fn main() {
    println!("{:?}", 1);
    println!("{:?}", 2.0f64);
    println!("{:?}", Numbers::Three);
    println!("{:?}", vec![4]);
}
