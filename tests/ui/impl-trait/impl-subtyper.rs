//@ check-pass

#![crate_type = "lib"]
fn checkpoints() -> impl Iterator {
    Some(()).iter().flat_map(|_| std::iter::once(()))
}

fn block_checkpoints() -> impl Iterator {
    checkpoints()
}

fn iter_raw() -> impl Iterator {
    let mut iter = block_checkpoints();

    (0..9).map(move |_| {
        iter.next();
    })
}
