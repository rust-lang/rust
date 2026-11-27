fn bar(_: Vec<i32>) {}
fn baz(_: &Vec<&i32>) {}
fn main() {
    let v = vec![&1];
    bar(v); //~ ERROR E0308
    let v = vec![];
    baz(&v);
    baz(&v);
    bar(v); //~ ERROR E0308
    let v = vec![];
    baz(&v);
    bar(v); //~ ERROR E0308
}
