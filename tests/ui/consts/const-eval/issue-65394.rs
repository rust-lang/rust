//@ revisions: stock precise_drops
//@ check-pass

// This test originated from #65394. We conservatively assumed that `x` is still `LiveDrop` even
// after it has been moved because a mutable reference to it exists at some point in the const body.
//
// With `&mut` in `const` being stable, this surprising behavior was observable.
#![cfg_attr(precise_drops, feature(const_precise_live_drops))]

const _: Vec<i32> = {
    let mut x = Vec::<i32>::new();
    let r = &mut x;
    let y = x;
    y
};

fn main() {}
