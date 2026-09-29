//@ edition: 2024..
//@ run-pass
fn test(x: &mut u8) -> impl FnMut() {
    || { let y: &mut u8 = x; *y += 1; }
}

fn main() {
    test(&mut 42)();
}
