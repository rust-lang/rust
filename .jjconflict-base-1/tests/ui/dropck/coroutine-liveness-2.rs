//@ check-pass
//@ edition: 2021

// regression test found while working on #117134.
use std::future;

fn main() {
    let mut recv = future::ready(());
    let _combined_fut = async {
        let _ = || read(&mut recv);
    };

    let _uwu = (String::new(), _combined_fut);
    // Dropping a coroutine as part of a more complex
    // types should not add unnecessary liveness
    // constraints.

    drop(recv);
}

fn read<F: future::Future>(_: &mut F) -> F::Output {
    todo!()
}
