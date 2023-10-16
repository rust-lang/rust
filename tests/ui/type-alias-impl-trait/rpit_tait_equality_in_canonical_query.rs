// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// check-pass

#![feature(type_alias_impl_trait)]

type Opaque = impl Sized;

fn get_rpit() -> impl Clone {}

fn query(_: impl FnOnce() -> Opaque) {}

fn test() -> Opaque {
    query(get_rpit);
    get_rpit()
}

fn main() {}
