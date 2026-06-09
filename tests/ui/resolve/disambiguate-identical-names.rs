pub mod submod {
    // Create ambiguity with the std::vec::Vec item:
    pub struct Vec;
}

fn test(_v: &Vec<Vec<u32>>) {
}

fn main() {
    let v = std::collections::HashMap::new();
    v.insert(3u8, 1u8);

    test(&v);
    //~^ ERROR mismatched types
}
