const fn foo(a: i32) -> Vec<i32> {
    vec![1, 2, 3] //~ ERROR heap allocations are not allowed in const fn
}

fn main() {}
