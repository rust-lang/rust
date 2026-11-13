fn foo<'a>((x, y): (&'a i32, &i32)) -> &'a i32 {
    if x > y { x } else { y } //~ ERROR explicit lifetime
}

fn main () { }
