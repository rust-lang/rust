//@compile-flags: -Zmiri-symbolic-alignment-check

extern "C" {
    static _dispatch_queue_attr_concurrent: [u8; 0];
}

static DISPATCH_QUEUE_CONCURRENT: &'static [u8; 0] = unsafe { &_dispatch_queue_attr_concurrent };

fn main() {
    let _val = *DISPATCH_QUEUE_CONCURRENT; //~ERROR: is not supported
}
