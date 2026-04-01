//@ run-rustfix
fn main() {
    let _ = Box:new("foo".to_string());
    //~^ ERROR path separator must be a double colon
}
