struct Foo {
    pub v: Vec<String>
}

fn main() {
    let f = Foo { v: Vec::new() };
    f.v.push("cat".to_string()); //~ ERROR cannot borrow
}


struct S {
    x: i32,
}
fn foo() {
    let s = S { x: 42 };
    s.x += 1; //~ ERROR cannot assign
}

fn bar(s: S) {
    s.x += 1; //~ ERROR cannot assign
}
