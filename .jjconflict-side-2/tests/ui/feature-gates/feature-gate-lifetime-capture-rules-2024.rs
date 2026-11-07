fn foo(x: &Vec<i32>) -> impl Sized {
    x
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
