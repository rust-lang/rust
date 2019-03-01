// This test managed to tickle a part of the compiler where a region
// representing a static scope (a call-site scope, to be precise) was
// leaking the where clause of a type underlying an `impl Trait`. This
// caused an ICE in spots where we assume that all regions must be
// region_vid's (unique representatives used in NLL) and then
// attempted to eagerly convert the region to its region_vid, which
// does not work for scopes which do not have region_vids (apart from
// the 'static region).
//
// This regression test is just meant to check that we do not ICE,
// regardless of whether NLL is turned on or not.

// revisions: ast nll
//[ast]compile-flags: -Z borrowck=ast
//[mig]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows

// don't worry about the --compare-mode=nll on this test.
// ignore-compare-mode-nll

pub struct AndThen<B, F>(B, F);
fn and_then<F, B>(_: F) -> AndThen<B, F> where F: FnOnce() -> B { unimplemented!() }

pub trait Trait { }
impl<B, F> Trait for AndThen<B, F> { }

pub struct JoinAll<I> where I: Iterator { _elem: std::marker::PhantomData<I::Item> }
pub fn join_all<I>(_i: I) -> JoinAll<I> where I: Iterator { unimplemented!() }

pub struct PollFn<F, T>(F, std::marker::PhantomData<fn () -> T>);
pub fn poll_fn<T, F>(_f: F) -> PollFn<F, T> where F: FnMut() -> T { unimplemented!() }

impl<B, I: Iterator, F> Iterator for Map<I, F> where F: FnMut(I::Item) -> B {
    type Item = B;
    fn next(&mut self) -> Option<B> { unimplemented!() }
}

struct Map<I, F> { iter: I, f: F }

fn main() { let _b: Box<Trait + Send> = Box::new(graphql()); }

fn graphql() -> impl Trait
{
    let local = ();
    let m = |_: ()| poll_fn(|| { local; });
    //[ast]~^   ERROR closure may outlive the current function, but it borrows `local`
    //[mig]~^^  ERROR closure may outlive the current function, but it borrows `local`
    //[nll]~^^^ ERROR closure may outlive the current function, but it borrows `local`
    let v = Map { iter: std::iter::once(()), f: m };
    let f = || join_all(v);
    and_then(f)
}
