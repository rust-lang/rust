//@ check-pass

fn takes_closure_of_array_3<F>(f: F) where F: Fn([i32; 3]) {
    f([1, 2, 3]);
}

fn takes_closure_of_array_3_apit(f: impl Fn([i32; 3])) {
    f([1, 2, 3]);
}

fn returns_closure_of_array_3() -> impl Fn([i32; 3]) {
    |_| {}
}

fn main() {
    takes_closure_of_array_3(returns_closure_of_array_3());
    takes_closure_of_array_3_apit(returns_closure_of_array_3());
}
