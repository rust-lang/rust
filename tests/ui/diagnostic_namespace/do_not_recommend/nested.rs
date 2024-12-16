//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Root {}
trait DontRecommend {}
trait Other {}
trait Child {}

#[diagnostic::do_not_recommend]
impl<T> Root for T where T: DontRecommend {}

impl<T> DontRecommend for T where T: Other {}

#[diagnostic::do_not_recommend]
impl<T> Other for T where T: Child {}

fn needs_root<T: Root>() {}

fn main() {
    needs_root::<()>();
    //~^ ERROR the trait bound `(): Root` is not satisfied
}
