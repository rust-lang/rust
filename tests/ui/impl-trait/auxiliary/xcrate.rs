// NOTE commented out due to issue #45994
//pub fn fourway_add(a: i32) -> impl Fn(i32) -> impl Fn(i32) -> impl Fn(i32) -> i32 {
//    move |b| move |c| move |d| a + b + c + d
//}

fn some_internal_fn() -> u32 {
    1
}

fn other_internal_fn() -> u32 {
    1
}

// See #40839
pub fn return_closure_accessing_internal_fn() -> impl Fn() -> u32 {
    || {
        some_internal_fn() + 1
    }
}

pub fn return_internal_fn() -> impl Fn() -> u32 {
    other_internal_fn
}
