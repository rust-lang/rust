//@ run-fail
//@ ignore-i686-pc-windows-msvc: #112480
//@ compile-flags: -C debug-assertions
//@ check-run-results

struct Misalignment {
    a: u32,
}

fn main() {
    let mut items: [Misalignment; 2] = [Misalignment { a: 0 }, Misalignment { a: 1 }];
    unsafe {
        let ptr: *const Misalignment = items.as_ptr().byte_add(1);
        let _ptr: &u32 = unsafe { &(*ptr).a };
    }
}
