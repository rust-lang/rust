//@ check-pass
//@ edition: 2021

// regression test for #116242.
use std::future;

fn main() {
    let mut recv = future::ready(());
    let _combined_fut = async {
        let _ = || read(&mut recv);
    };

    drop(recv);
}

fn read<F: future::Future>(_: &mut F) -> F::Output {
    todo!()
}
