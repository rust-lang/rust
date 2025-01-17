// Combination of `ptr-to-trait-obj-wrap.rs` and `ptr-to-trait-obj-add-auto.rs`.
//
// Checks that you *can't* add auto traits to trait object in pointer casts involving wrapping said
// traits structures.
#![allow(unused)]
#![deny(ptr_cast_add_auto_to_object)]

trait A {}

struct W<T: ?Sized>(T);
struct X<T: ?Sized>(T);

fn unwrap(a: *const W<dyn A>) -> *const (dyn A + Send) {
    a as _
    //~^ error: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn unwrap_nested(a: *const W<W<dyn A>>) -> *const W<dyn A + Send> {
    a as _
    //~^ error: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn rewrap(a: *const W<dyn A>) -> *const X<dyn A + Send> {
    a as _
    //~^ error: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn rewrap_nested(a: *const W<W<dyn A>>) -> *const W<X<dyn A + Send>> {
    a as _
    //~^ error: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn wrap(a: *const dyn A) -> *const W<dyn A + Send> {
    a as _
    //~^ error: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
