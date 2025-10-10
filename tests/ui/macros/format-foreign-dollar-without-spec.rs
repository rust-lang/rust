// https://github.com/rust-lang/rust/issues/137580
fn main() {
    println!("%65536$", 1);
    //~^ ERROR never used
}
