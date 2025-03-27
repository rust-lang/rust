//@ run-pass
#![allow(unused_variables)]
#![allow(unnecessary_refs)]
fn gimme_a_raw_pointer<T>(_: *const T) { }

fn test<T>(t: T) { }

fn main() {
    // Clearly `pointer` must be of type `*const ()`.
    let pointer = &() as *const _;
    gimme_a_raw_pointer(pointer);

    let t = test as fn (i32);
    t(0i32);
}
