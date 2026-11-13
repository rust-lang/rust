//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.intro

trait Root {}
trait DontRecommend {}
trait Other {}

#[diagnostic::do_not_recommend]
impl<T> Root for T where T: DontRecommend {}

impl<T> DontRecommend for T where T: Other {}

fn needs_root<T: Root>() {}

fn main() {
    needs_root::<()>();
    //~^ ERROR the trait bound `(): Root` is not satisfied
}
