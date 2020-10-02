// run-rustfix
fn main() {
    let _: usize = std::mem:size_of::<u32>();
    //~^ ERROR casts cannot be followed by a function call
}
