// Test that we give suitable error messages when the user attempts to
// impl a trait `Trait` for its own object type.

trait Foo { fn dummy(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

// Supertraits of Baz are not legal:
impl Foo for Baz { }   //~ ERROR E0371
impl Bar for Baz { }   //~ ERROR E0371
impl Baz for Baz { }   //~ ERROR E0371

// But other random traits are:
trait Other { }
impl Other for Baz { } // OK, Other not a supertrait of Baz

fn main() { }
