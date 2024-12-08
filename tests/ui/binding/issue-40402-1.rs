// Check that we do not suggest `ref f` here in the `main()` function.
struct Foo {
    pub v: Vec<String>,
}

fn main() {
    let mut f = Foo { v: Vec::new() };
    f.v.push("hello".to_string());
    let e = f.v[0]; //~ ERROR cannot move out of index
}
