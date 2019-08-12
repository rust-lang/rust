fn main() {
    let x = || {
        Err(())?; //~ ERROR type annotations needed for the closure
        Ok(())
    };
}
