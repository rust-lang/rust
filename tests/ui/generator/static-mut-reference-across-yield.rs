// build-pass
// revisions: mir thir drop_tracking drop_tracking_mir
// [thir]compile-flags: -Zthir-unsafeck
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

#![feature(generators)]

static mut A: [i32; 5] = [1, 2, 3, 4, 5];

fn is_send_sync<T: Send + Sync>(_: T) {}

fn main() {
    unsafe {
        let gen_index = static || {
            let u = A[{
                yield;
                1
            }];
        };
        let gen_match = static || match A {
            i if {
                yield;
                true
            } =>
            {
                ()
            }
            _ => (),
        };
        is_send_sync(gen_index);
        is_send_sync(gen_match);
    }
}
