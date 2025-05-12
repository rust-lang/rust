#![feature(auto_traits)]
#![feature(negative_impls)]

// Test for issue #56934 - that it is impossible to redundantly
// implement an auto-trait for a trait object type that contains it.

// Negative impl variant.

auto trait Marker1 {}
auto trait Marker2 {}

trait Object: Marker1 {}

// A supertrait marker is illegal...
impl !Marker1 for dyn Object + Marker2 {} //~ ERROR E0371
                                          //~^ ERROR 0321
// ...and also a direct component.
impl !Marker2 for dyn Object + Marker2 {} //~ ERROR E0371
                                          //~^ ERROR 0321

// A non-principal trait-object type is orphan even in its crate.
impl !Send for dyn Marker2 {} //~ ERROR E0117

// Implementing a marker for a local trait object is forbidden by a special
// orphan-like rule.
impl !Marker2 for dyn Object {} //~ ERROR E0321
impl !Send for dyn Object {} //~ ERROR E0321
impl !Send for dyn Object + Marker2 {} //~ ERROR E0321

// Blanket impl that applies to dyn Object is equally problematic.
auto trait Marker3 {}
impl<T: ?Sized> !Marker3 for T {} //~ ERROR E0321

auto trait Marker4 {}
impl<T> !Marker4 for T {} // okay

fn main() {}
