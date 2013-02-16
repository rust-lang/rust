// compile-flags: --test

// Using uv_ll, cell, net::ip
extern mod std;

// Some basic logging
fn macros() {
    macro_rules! rtdebug_ (
        ($( $arg:expr),+) => ( {
            dumb_println(fmt!( $($arg),+ ));

            fn dumb_println(s: &str) {
                use core::str::as_c_str;
                use core::libc::c_char;

                extern {
                    fn printf(s: *c_char);
                }

                do as_c_str(s.to_str() + "\n") |s| {
                    unsafe { printf(s); }
                }
            }

        } )
    )

    // An alternate version with no output, for turning off logging
    macro_rules! rtdebug (
        ($( $arg:expr),+) => ( { } )
    )
}

// FIXME #4981: Wish I would write these `mod sched #[path = "newrt_sched.rs"];`
#[path = "newrt_sched.rs"] mod sched;
#[path = "newrt_io.rs"] mod io;
#[path = "newrt_uvio.rs"] mod uvio;
#[path = "newrt_uv.rs"] mod uv;
// FIXME: The import in `sched` doesn't resolve unless this is pub!
#[path = "newrt_thread_local_storage.rs"] pub mod thread_local_storage;
#[path = "newrt_work_queue.rs"] mod work_queue;
#[path = "newrt_stack.rs"] mod stack;
#[path = "newrt_context.rs"] mod context;
#[path = "newrt_thread.rs"] mod thread;
