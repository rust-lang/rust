fn main() {
    // With the nested Vec, this is calling Offset(Unique::empty(), 0) on drop.
    let args: Vec<Vec<i32>> = Vec::new();
    let _val = Box::new(args);
}
