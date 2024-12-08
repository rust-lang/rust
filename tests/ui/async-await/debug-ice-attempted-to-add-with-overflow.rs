//@ check-fail
//@ edition:2021

// test for issue-114912 - debug ice: attempted to add with overflow

async fn main() {
    //~^ ERROR `main` function is not allowed to be `async`
    [0usize; 0xffff_ffff_ffff_ffff].await;
    //~^ ERROR `[usize; usize::MAX]` is not a future
}
