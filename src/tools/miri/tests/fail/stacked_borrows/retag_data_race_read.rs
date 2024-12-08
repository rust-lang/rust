//! Make sure that a retag acts like a read for the data race model.
// Avoid accidental synchronization via address reuse.
//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-address-reuse-cross-thread-rate=0
#[derive(Copy, Clone)]
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

fn thread_1(p: SendPtr) {
    let p = p.0;
    unsafe {
        let _r = &*p;
    }
}

fn thread_2(p: SendPtr) {
    let p = p.0;
    unsafe {
        *p = 5; //~ ERROR: Data race detected between (1) retag read on thread `unnamed-1` and (2) non-atomic write on thread `unnamed-2`
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
