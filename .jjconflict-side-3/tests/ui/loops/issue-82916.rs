struct S(i32);

fn foo(x: Vec<S>) {
    for y in x {

    }
    let z = x; //~ ERROR use of moved value: `x`
}

fn main() {}
