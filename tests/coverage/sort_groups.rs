//@ edition: 2021

// Demonstrate that `sort_subviews.py` can sort instantiation groups into a
// predictable order, while preserving their heterogeneous contents.

fn main() {
    let cond = std::env::args().len() > 1;
    generic_fn::<()>(cond);
    generic_fn::<&'static str>(!cond);
    if std::hint::black_box(false) {
        generic_fn::<char>(cond);
    }
    generic_fn::<i32>(cond);
    other_fn();
}

fn generic_fn<T>(cond: bool) {
    if cond {
        println!("{}", std::any::type_name::<T>());
    }
}

fn other_fn() {}
