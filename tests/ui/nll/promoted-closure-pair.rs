// Check that we handle multiple closures in the same promoted constant.

fn foo() -> &'static i32 {
    let z = 0;
    let p = &(|y| y, |y| y);
    p.0(&z);
    p.1(&z)         //~ ERROR cannot return
}

fn main() {}
