//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.intro

trait Root {}
trait DontRecommend {}

impl<T> Root for T where T: DontRecommend {}

// this has no effect yet for resolving the trait error below
#[diagnostic::do_not_recommend]
impl<T> DontRecommend for &'static T {}

fn needs_root<T: Root>() {}

fn foo<'a>(a: &'a ()) {
    needs_root::<&'a ()>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    foo(&());
}
