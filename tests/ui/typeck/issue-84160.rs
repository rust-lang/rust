fn mismatched_types_with_reference(x: &u32) -> &u32 {
    if false {
        return x;
    }
    return "test";
    //~^ERROR mismatched types
}

fn main() {}
