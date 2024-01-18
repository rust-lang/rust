// This should not pass, because `usize: Fsm` does not hold. However, it currently ICEs.

// check-fail
// known-bug: #80409
// failure-status: 101
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
// normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
// rustc-env:RUST_BACKTRACE=0

#![allow(unreachable_code, unused)]

use std::marker::PhantomData;

struct FsmBuilder<TFsm> {
    _fsm: PhantomData<TFsm>,
}

impl<TFsm> FsmBuilder<TFsm> {
    fn state(&mut self) -> FsmStateBuilder<TFsm> {
        todo!()
    }
}

struct FsmStateBuilder<TFsm> {
    _state: PhantomData<TFsm>,
}

impl<TFsm> FsmStateBuilder<TFsm> {
    fn on_entry<TAction: Fn(&mut StateContext<'_, TFsm>)>(&self, _action: TAction) {}
}

trait Fsm {
    type Context;
}

struct StateContext<'a, TFsm: Fsm> {
    context: &'a mut TFsm::Context,
}

fn main() {
    let mut builder: FsmBuilder<usize> = todo!();
    builder.state().on_entry(|_| {});
}
