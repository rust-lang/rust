//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2021

// This is a regression test for an ICE.
//
// This results in a query cycle with the new solver. Let's not
// bother with this when stabilizing the new solver. For more context,
// see https://rust-lang.zulipchat.com/#narrow/channel/364551-t-types.2Ftrait-system-refactor/topic/dyn.20compatibility.20check.20in.20object.20candidate.20causes.20cycle/with/627370998


trait Foo {
    //[next]~^ ERROR: cycle detected when checking if trait `Foo` is dyn-compatible
    async fn foo(self: &dyn Foo) {
        //[old]~^ ERROR: `Foo` is not dyn compatible
        //[old]~| ERROR: invalid `self` parameter type: `&dyn Foo`
        todo!()
    }
}

fn main() {}
