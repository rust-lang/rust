// check-pass

#![feature(marker_trait_attr)]

trait NonMarker {}
#[marker]
trait Marker {}

fn main() {
    let _: &(dyn NonMarker + Marker);
}
