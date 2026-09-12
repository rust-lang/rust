//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Unlike normal traits, marker traits are allowed to have overlapping
// impls, so we merge multiple applicable impl candidates for them.

#![feature(marker_trait_attr)]

trait Local {}

#[marker]
trait Marker {}
impl<T> Marker for Option<T> where Self: Clone {}
impl<T> Marker for Option<T> where Self: Local {}

fn impls_marker<T: Marker>() {}

fn test<T>()
where
    Option<T>: Clone + Local,
{
    impls_marker::<Option<T>>();
}

fn main() {}
