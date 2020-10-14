const fn foo(a: i32) -> Vec<i32> {
    vec![1, 2, 3]
    //~^ ERROR allocations are not allowed
    //~| ERROR calls in constant functions
}

fn main() {}
