fn main() {
    let x = |a: (), b: ()| {
        Err(a)?; //~ ERROR type annotations needed for the closure
        Ok(b)
    };
}
