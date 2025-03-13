// Combination of `ptr-to-trait-obj-wrap.rs` and `ptr-to-trait-obj-add-auto.rs`.
//
// Checks that you *can't* add auto traits to trait object in pointer casts involving wrapping said
// traits structures.

trait A {}

struct W<T: ?Sized>(T);
struct X<T: ?Sized>(T);

fn unwrap(a: *const W<dyn A>) -> *const (dyn A + Send) {
    a as _
    //~^ error: cannot add auto trait `Send` to dyn bound via pointer cast
}

fn unwrap_nested(a: *const W<W<dyn A>>) -> *const W<dyn A + Send> {
    a as _
    //~^ error: cannot add auto trait `Send` to dyn bound via pointer cast
}

fn rewrap(a: *const W<dyn A>) -> *const X<dyn A + Send> {
    a as _
    //~^ error: cannot add auto trait `Send` to dyn bound via pointer cast
}

fn rewrap_nested(a: *const W<W<dyn A>>) -> *const W<X<dyn A + Send>> {
    a as _
    //~^ error: cannot add auto trait `Send` to dyn bound via pointer cast
}

fn wrap(a: *const dyn A) -> *const W<dyn A + Send> {
    a as _
    //~^ error: cannot add auto trait `Send` to dyn bound via pointer cast
}

fn main() {}
