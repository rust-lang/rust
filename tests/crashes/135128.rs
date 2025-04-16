//@ known-bug: #135128
//@ compile-flags: -Copt-level=1
//@ edition: 2021

#![feature(trivial_bounds)]

async fn return_str() -> str
where
    str: Sized,
{
    *"Sized".to_string().into_boxed_str()
}
fn main() {}
