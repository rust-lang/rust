//@ run-rustfix
fn main() {
    let _ = Option:Some("");
    //~^ ERROR path separator must be a double colon
}
