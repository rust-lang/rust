#![feature(rustc_attrs)]
#![feature(negative_impls)]

// Test for issue #56934 - that it is impossible to redundantly
// implement an auto-trait for a trait object type that contains it.

// Positive impl variant.

#[rustc_auto_trait]
trait Marker1 {}
#[rustc_auto_trait]
trait Marker2 {}

trait Object: Marker1 {}

// A supertrait marker is illegal...
impl Marker1 for dyn Object + Marker2 {} //~ ERROR E0371
                                         //~^ ERROR E0321
// ...and also a direct component.
impl Marker2 for dyn Object + Marker2 {} //~ ERROR E0371
                                         //~^ ERROR E0321

// A non-principal trait-object type is orphan even in its crate.
unsafe impl Send for dyn Marker2 {} //~ ERROR E0117

// Implementing a marker for a local trait object is forbidden by a special
// orphan-like rule.
impl Marker2 for dyn Object {} //~ ERROR E0321
unsafe impl Send for dyn Object {} //~ ERROR E0321
unsafe impl Send for dyn Object + Marker2 {} //~ ERROR E0321

// Blanket impl that applies to dyn Object is equally problematic.
#[rustc_auto_trait]
trait Marker3 {}
impl<T: ?Sized> Marker3 for T {} //~ ERROR E0321

#[rustc_auto_trait]
trait Marker4 {}
impl<T> Marker4 for T {} // okay

fn main() {}
