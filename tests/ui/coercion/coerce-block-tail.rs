//@ check-fail
fn main() {
    let _: &str = & { String::from("hahah")};
    let _: &i32 = & { Box::new(1i32) };
    //~^ ERROR mismatched types
}
