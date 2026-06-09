//@ build-pass

#![feature(coroutines, stmt_expr_attributes)]
// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static mut A: [i32; 5] = [1, 2, 3, 4, 5];

fn is_send_sync<T: Send + Sync>(_: T) {}

fn main() {
    unsafe {
        let gen_index = #[coroutine]
        static || {
            let u = A[{
                yield;
                1
            }];
        };
        let gen_match = #[coroutine]
        static || match A {
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
