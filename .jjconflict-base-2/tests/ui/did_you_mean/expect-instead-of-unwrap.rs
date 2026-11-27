fn main() {
    Ok(42).unwrap("wow");
    //~^ ERROR this method takes 0 arguments but 1 argument was supplied
    Some(42).unwrap("wow");
    //~^ ERROR this method takes 0 arguments but 1 argument was supplied
}
