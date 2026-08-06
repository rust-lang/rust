// Checks that we don't get conflicting arguments in our debug info with a particular async function
// structure.

//@ edition:2021
//@ compile-flags: -Cdebuginfo=2
//@ build-pass

#![crate_type = "lib"]

use std::future::Future;

// The compiler produces a closure as part of this function. That closure initially takes an
// argument _task_context. Later, when the MIR for that closure is transformed into a coroutine
// state machine, _task_context is demoted to not be an argument, but just part of an unnamed
// argument. If we emit debug info saying that both _task_context and the unnamed argument are both
// argument number 2, then LLVM will fail with "conflicting debug info for argument". See
// https://github.com/rust-lang/rust/pull/109466#issuecomment-1500879195 for details.
async fn recv_unit() {
    std::future::ready(()).await;
}

pub fn poll_recv() {
    // This box is necessary in order to reproduce the problem.
    let _: Box<dyn Future<Output = ()>> = Box::new(recv_unit());
}
