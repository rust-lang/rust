//@ revisions: stock precise_drops
//@[precise_drops] check-pass

// This test originated from #65394. We conservatively assume that `x` is still `LiveDrop` even
// after it has been moved because a mutable reference to it exists at some point in the const body.
//
// With `&mut` in `const` being stable, this surprising behavior is now observable.
// `const_precise_live_drops` fixes that.
#![cfg_attr(precise_drops, feature(const_precise_live_drops))]

const _: Vec<i32> = {
    let mut x = Vec::<i32>::new(); //[stock]~ ERROR destructor of
    let r = &mut x;
    let y = x;
    y
};

fn main() {}
