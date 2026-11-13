//@ edition: 2024
//@ check-pass
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// We should support calculating alias liveness with const parameters.

fn consumed<'o, I: Iterator<Item = u32>, const N: usize>(
    iter: I,
) -> impl Iterator<Item = u32> + 'o {
    iter.collect::<Vec<_>>().into_iter()
}

fn main() {
    let mut cells = vec![1u32, 2, 3];
    let it = consumed::<_, 0>(cells.iter_mut().map(|c| *c));
    cells[0] = 7;
    drop(it);
}
