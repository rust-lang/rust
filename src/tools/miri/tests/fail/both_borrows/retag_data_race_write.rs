//! Make sure that a retag acts like a write for the data race model.
//@revisions: stack tree
//@compile-flags: -Zmiri-deterministic-concurrency
//@[tree]compile-flags: -Zmiri-tree-borrows
#[derive(Copy, Clone)]
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

fn thread_1(p: SendPtr) {
    let p = p.0;
    unsafe {
        let _r = &mut *p;
    }
}

fn thread_2(p: SendPtr) {
    let p = p.0;
    unsafe {
        *p = 5; //~ ERROR: /Data race detected between \(1\) retag (read|write) on thread `unnamed-[0-9]+` and \(2\) non-atomic write on thread `unnamed-[0-9]+`/
    }
}

fn main() {
    let mut x = 0;
    let p = std::ptr::addr_of_mut!(x);
    let p = SendPtr(p);

    let t1 = std::thread::spawn(move || thread_1(p));
    let t2 = std::thread::spawn(move || thread_2(p));
    let _ = t1.join();
    let _ = t2.join();
}
