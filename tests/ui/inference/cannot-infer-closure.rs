fn main() {
    let x = |a: (), b: ()| {
        Err(a)?;
        Ok(b)
        //~^ ERROR type annotations needed
    };
}
