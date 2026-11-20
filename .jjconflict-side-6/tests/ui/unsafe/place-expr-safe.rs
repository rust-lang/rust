//@ check-pass

fn main() {
    let ptr = std::ptr::null_mut::<i32>();
    let addr = &raw const *ptr;

    let local = 1;
    let ptr = &local as *const i32;
    let addr = &raw const *ptr;

    let boxed = Box::new(1);
    let ptr = &*boxed as *const i32;
    let addr = &raw const *ptr;
}
