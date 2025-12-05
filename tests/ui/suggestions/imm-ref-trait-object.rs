fn test(t: &dyn Iterator<Item=&u64>) -> u64 {
     t.min().unwrap() //~ ERROR the `min` method cannot be invoked on `&dyn Iterator<Item = &u64>`
}

fn main() {
     let array = [0u64];
     test(&mut array.iter());
}
