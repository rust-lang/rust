fn deref_int(a: &i32) -> i32 {
    *a
}

fn main() {
    // https://github.com/rust-lang/rust/issues/112293
    let _has_inference_vars: Option<i32> = Some(0).map(deref_int);
    //~^ ERROR type mismatch in function arguments
}
