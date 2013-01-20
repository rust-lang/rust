// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A process-wide libuv event loop for library use.

#[forbid(deprecated_mode)];

use ll = uv_ll;
use iotask = uv_iotask;
use get_gl = get;
use uv_iotask::{IoTask, spawn_iotask};

use core::either::{Left, Right};
use core::libc;
use core::pipes::{Port, Chan, SharedChan, select2i};
use core::private::global::{global_data_clone_create,
                            global_data_clone};
use core::private::weak_task::weaken_task;
use core::str;
use core::task::{task, SingleThreaded, spawn};
use core::task;
use core::vec;
use core::clone::Clone;
use core::option::{Some, None};

/**
 * Race-free helper to get access to a global task where a libuv
 * loop is running.
 *
 * Use `uv::hl::interact` to do operations against the global
 * loop that this function returns.
 *
 * # Return
 *
 * * A `hl::high_level_loop` that encapsulates communication with the global
 * loop.
 */
pub fn get() -> IoTask {
    return get_monitor_task_gl();
}

#[doc(hidden)]
fn get_monitor_task_gl() -> IoTask unsafe {

    type MonChan = Chan<IoTask>;

    struct GlobalIoTask(IoTask);

    impl GlobalIoTask: Clone {
        fn clone(&self) -> GlobalIoTask {
            GlobalIoTask((**self).clone())
        }
    }

    fn key(_: GlobalIoTask) { }

    match global_data_clone(key) {
        Some(GlobalIoTask(iotask)) => iotask,
        None => {
            let iotask: IoTask = spawn_loop();
            let mut installed = false;
            let final_iotask = do global_data_clone_create(key) {
                installed = true;
                ~GlobalIoTask(iotask.clone())
            };
            if installed {
                do task().unlinked().spawn() unsafe {
                    debug!("global monitor task starting");
                    // As a weak task the runtime will notify us when to exit
                    do weaken_task |weak_exit_po| {
                        debug!("global monitor task is now weak");
                        weak_exit_po.recv();
                        iotask::exit(&iotask);
                        debug!("global monitor task is leaving weakend state");
                    };
                    debug!("global monitor task exiting");
                }
            } else {
                iotask::exit(&iotask);
            }

            match final_iotask {
                GlobalIoTask(iotask) => iotask
            }
        }
    }
}

fn spawn_loop() -> IoTask {
    let builder = do task().add_wrapper |task_body| {
        fn~(move task_body) {
            // The I/O loop task also needs to be weak so it doesn't keep
            // the runtime alive
            unsafe {
                do weaken_task |_| {
                    debug!("global libuv task is now weak");
                    task_body();

                    // We don't wait for the exit message on weak_exit_po
                    // because the monitor task will tell the uv loop when to
                    // exit

                    debug!("global libuv task is leaving weakened state");
                }
            }
        }
    };
    let builder = builder.unlinked();
    spawn_iotask(move builder)
}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use uv::iotask;
    use uv::ll;
    use uv_global_loop::*;

    use core::iter;
    use core::libc;
    use core::oldcomm;
    use core::ptr;
    use core::task;

    extern fn simple_timer_close_cb(timer_ptr: *ll::uv_timer_t) unsafe {
        let exit_ch_ptr = ll::get_data_for_uv_handle(
            timer_ptr as *libc::c_void) as *oldcomm::Chan<bool>;
        let exit_ch = *exit_ch_ptr;
        oldcomm::send(exit_ch, true);
        log(debug, fmt!("EXIT_CH_PTR simple_timer_close_cb exit_ch_ptr: %?",
                       exit_ch_ptr));
    }
    extern fn simple_timer_cb(timer_ptr: *ll::uv_timer_t,
                             _status: libc::c_int) unsafe {
        log(debug, ~"in simple timer cb");
        ll::timer_stop(timer_ptr);
        let hl_loop = &get_gl();
        do iotask::interact(hl_loop) |_loop_ptr| unsafe {
            log(debug, ~"closing timer");
            ll::close(timer_ptr, simple_timer_close_cb);
            log(debug, ~"about to deref exit_ch_ptr");
            log(debug, ~"after msg sent on deref'd exit_ch");
        };
        log(debug, ~"exiting simple timer cb");
    }

    fn impl_uv_hl_simple_timer(iotask: &IoTask) unsafe {
        let exit_po = oldcomm::Port::<bool>();
        let exit_ch = oldcomm::Chan(&exit_po);
        let exit_ch_ptr = ptr::addr_of(&exit_ch);
        log(debug, fmt!("EXIT_CH_PTR newly created exit_ch_ptr: %?",
                       exit_ch_ptr));
        let timer_handle = ll::timer_t();
        let timer_ptr = ptr::addr_of(&timer_handle);
        do iotask::interact(iotask) |loop_ptr| unsafe {
            log(debug, ~"user code inside interact loop!!!");
            let init_status = ll::timer_init(loop_ptr, timer_ptr);
            if(init_status == 0i32) {
                ll::set_data_for_uv_handle(
                    timer_ptr as *libc::c_void,
                    exit_ch_ptr as *libc::c_void);
                let start_status = ll::timer_start(timer_ptr, simple_timer_cb,
                                                   1u, 0u);
                if(start_status == 0i32) {
                }
                else {
                    fail ~"failure on ll::timer_start()";
                }
            }
            else {
                fail ~"failure on ll::timer_init()";
            }
        };
        oldcomm::recv(exit_po);
        log(debug, ~"global_loop timer test: msg recv on exit_po, done..");
    }

    #[test]
    fn test_gl_uv_global_loop_high_level_global_timer() unsafe {
        let hl_loop = &get_gl();
        let exit_po = oldcomm::Port::<()>();
        let exit_ch = oldcomm::Chan(&exit_po);
        task::spawn_sched(task::ManualThreads(1u), || {
            let hl_loop = &get_gl();
            impl_uv_hl_simple_timer(hl_loop);
            oldcomm::send(exit_ch, ());
        });
        impl_uv_hl_simple_timer(hl_loop);
        oldcomm::recv(exit_po);
    }

    // keeping this test ignored until some kind of stress-test-harness
    // is set up for the build bots
    #[test]
    #[ignore]
    fn test_stress_gl_uv_global_loop_high_level_global_timer() unsafe {
        let exit_po = oldcomm::Port::<()>();
        let exit_ch = oldcomm::Chan(&exit_po);
        let cycles = 5000u;
        for iter::repeat(cycles) {
            task::spawn_sched(task::ManualThreads(1u), || {
                let hl_loop = &get_gl();
                impl_uv_hl_simple_timer(hl_loop);
                oldcomm::send(exit_ch, ());
            });
        };
        for iter::repeat(cycles) {
            oldcomm::recv(exit_po);
        };
        log(debug, ~"test_stress_gl_uv_global_loop_high_level_global_timer"+
            ~" exiting sucessfully!");
    }
}
