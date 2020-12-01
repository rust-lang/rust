// run-rustfix
// Tests from for_loop.rs that don't have suggestions

#[warn(clippy::single_element_loop)]
fn main() {
    let item1 = 2;
    for item in &[item1] {
        println!("{}", item);
    }
}
