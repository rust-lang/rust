struct Foo {
    field: i32,
}

fn foo2<'a>(a: &'a Foo, x: &i32) -> &'a i32 {
    if true {
        let p: &i32 = &a.field;
        &*p
    } else {
        &*x //~ ERROR explicit lifetime
    }
}

fn main() { }
