fn bar() {}
fn foo(x: i32) -> u32 {
    0
}
fn main() {
    let b: fn() -> u32 = bar; //~ ERROR mismatched types [E0308]
    let f: fn(i32) = foo; //~ ERROR mismatched types [E0308]
}
