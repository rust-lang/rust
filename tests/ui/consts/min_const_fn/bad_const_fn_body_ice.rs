fn non_const() -> Vec<i32> {
    vec![1, 2, 3]
}

const fn foo(a: i32) -> Vec<i32> {
    non_const()
    //~^ ERROR cannot call non-const function
}

fn main() {}
