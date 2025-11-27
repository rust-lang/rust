#![feature(reborrow)]
use std::ops::{CoerceShared, Reborrow};
use std::marker::PhantomData;

#[derive(Debug)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
#[derive(Debug, Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

impl<'a> CoerceShared<CustomMarkerRef<'a>> for CustomMarker<'a> {}

impl<'a> From<CustomMarker<'a>> for CustomMarkerRef<'a> {
    fn from(value: CustomMarker<'a>) -> Self {
        Self(PhantomData)
    }
}

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn method_two<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn main() {
    let mut a = CustomMarker(PhantomData);
    let b = method(a);
    // let c = method(a); // should invalidate b
    let c = method_two(a); // should invalidate b
    println!("{c:?} {b:?} {a:?}");
}

// fn main_using_normal_references() {
//     let a = &mut ();
//     let b = method(a);
//     let _ = method(a);
//     eprintln!("{b}");
// }
