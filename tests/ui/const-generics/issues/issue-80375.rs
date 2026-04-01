struct MyArray<const COUNT: usize>([u8; COUNT + 1]);
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
