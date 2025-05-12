trait User{
    fn test();
    /// empty doc
    //~^ ERROR found a documentation comment that doesn't document anything
}
fn main() {}
