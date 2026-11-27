fn foo<'a>(x: &i32, y: &'a i32) -> &'a i32 {
    if x > y { x } else { y } //~ ERROR explicit lifetime
}

fn main() { }
