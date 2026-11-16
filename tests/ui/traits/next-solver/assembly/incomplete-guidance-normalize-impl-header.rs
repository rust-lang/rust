//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #138707
//@edition:2024
//@compile-flags: --crate-type lib

// When using the `From` impl we eagerly normalize
// `<DefaultAllocator as Allocator<?r>>::Buffer` before
// ever equating the impl header with the goal. Because
// of the `DefaultAllocator: Allocator<D>`, this keeps the
// alias as rigid without actually constraining `?r` to `D`
// in the old solver.
//
// This then passes HIR typeck as constraining the `?s` of
// `LeftReflector<?s>` to an alias is fine and that alias gets
// later normalized to `()`. However, now keeping the alias in
// the impl header as rigid results in a failure when relating
// the impl header with `LeftReflector<()>` from the goal.
//
// This only passes with the new solver as we currently don't
// explicitly normalize the impl header before equating it with
// the goal, instead replacing non-rigid aliases with inference
// variables on demand. We may end up breaking this test if we
// start to eagerly normalize within the new trait solver.

struct LeftReflector<S>(S);
struct DefaultAllocator {}

trait Allocator<R> {
    type Buffer;
}

impl Allocator<()> for DefaultAllocator {
    type Buffer = ();
}


impl<R> From<R> for LeftReflector<<DefaultAllocator as Allocator<R>>::Buffer>
where
    DefaultAllocator: Allocator<R>,
{
    fn from(_: R) -> Self {
        todo!()
    }
}

fn ice<D>(a: ())
where
    DefaultAllocator: Allocator<D>,
{
    // ICE
    let _ = LeftReflector::from(a);
}
