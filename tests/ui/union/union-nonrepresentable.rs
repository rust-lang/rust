union U { //~ ERROR recursive type `U` has infinite size
    a: u8,
    b: std::mem::ManuallyDrop<U>,
}

fn main() {}
