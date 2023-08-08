// revisions: good1 good2 good3
// check-pass

use std::ops::Drop;

struct Foo;

const X: usize = 1;

#[cfg(good1)]
impl Drop for Foo
where
    [(); X]:, // Trivial WF bound
{
    fn drop(&mut self) {}
}

#[cfg(good2)]
impl Drop for Foo
where
    for<'a> &'a (): Copy, // Trivial trait bound
{
    fn drop(&mut self) {}
}

#[cfg(good3)]
impl Drop for Foo
where
    for<'a> &'a (): 'a, // Trivial outlives bound
{
    fn drop(&mut self) {}
}

fn main() {}
