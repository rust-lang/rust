fn main() {
    let x = |a: (), b: ()| {
        Err::<(), _>(a)?;
        Ok(b)
        //~^ ERROR type annotations needed
    };
}
