#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a, 'b>(PhantomData<&'a mut ()>, PhantomData<&'b ()>);
impl<'a, 'b> Reborrow for CustomMarker<'a, 'b> {}
//~^ ERROR: implementing `Reborrow` requires that a single lifetime parameter is passed between source and target

fn main() {}
