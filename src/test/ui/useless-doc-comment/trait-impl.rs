//! issue #56766
//!
//! invalid doc comment at the end of a trait declaration

trait Foo {
    ///
    fn foo(&self);
    /// I am not documenting anything
    //~^ ERROR: found a documentation comment that doesn't document anything [E0585]
}

fn main() {

}
