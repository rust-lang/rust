// revisions: good1 good2 bad1 bad2
//[good1] check-pass
//[good2] check-pass

use std::ops::Drop;

struct DropMe<T: Copy>(T);

#[cfg(good1)]
impl<T> Drop for DropMe<T>
where
    T: Copy + Clone,
{
    fn drop(&mut self) {}
}

#[cfg(good2)]
impl<T> Drop for DropMe<T>
where
    T: Copy,
    [T; 1]: Copy, // Trivial bound implied by `T: Copy`
{
    fn drop(&mut self) {}
}

#[cfg(bad1)]
impl<T> Drop for DropMe<T>
//[bad1]~^ ERROR the trait bound `T: Copy` is not satisfied
where
    [T; 1]: Copy, // But `[T; 1]: Copy` does not imply `T: Copy`
{
    fn drop(&mut self) {}
    //[bad1]~^ ERROR the trait bound `T: Copy` is not satisfied
}

#[cfg(bad2)]
impl<T> Drop for DropMe<T>
//[bad2]~^ ERROR the trait bound `T: Copy` is not satisfied
{
    fn drop(&mut self) {}
    //[bad2]~^ ERROR the trait bound `T: Copy` is not satisfied
}

fn main() {}
