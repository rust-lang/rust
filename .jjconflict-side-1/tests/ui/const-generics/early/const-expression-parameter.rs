fn i32_identity<const X: i32>() -> i32 {
    5
}

fn foo_a() {
    i32_identity::<-1>(); // ok
}

fn foo_b() {
    i32_identity::<1 + 2>(); //~ ERROR expressions must be enclosed in braces
}

fn foo_c() {
    i32_identity::< -1 >(); // ok
}

fn main() {
    i32_identity::<5>(); // ok
}
