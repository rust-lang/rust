// Regression test for #48697

fn foo(x: &i32) -> &i32 {
    let z = 4;
    let f = &|y| y;
    let k = f(&z);
    f(x) //~ ERROR cannot return value referencing local variable
}

fn main() {}
