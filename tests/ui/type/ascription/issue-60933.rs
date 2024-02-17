//@ run-rustfix
fn main() {
    let _: usize = std::mem:size_of::<u32>();
    //~^ ERROR path separator must be a double colon
}
