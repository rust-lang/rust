//@ run-pass

use std::sync::atomic::*;

static FLAG: AtomicBool = AtomicBool::new(false);

struct NoisyDrop(#[allow(dead_code)] &'static str);
impl Drop for NoisyDrop {
    fn drop(&mut self) {
        FLAG.store(true, Ordering::SeqCst);
    }
}
fn main() {
    {
        let _val = &&(NoisyDrop("drop!"), 0).1;
    }
    assert!(FLAG.load(Ordering::SeqCst));
}
