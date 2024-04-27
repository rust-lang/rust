//@ run-rustfix
fn main() {
    let _ = (-10..=10).find(|x: i32| x.signum() == 0); //~ ERROR type mismatch in closure arguments
    let _ = (-10..=10).find(|x: &&&i32| x.signum() == 0); //~ ERROR type mismatch in closure arguments
}
