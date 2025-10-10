// Test various places where relaxed bounds are not permitted.
//
// Relaxed bounds are only permitted inside impl-Trait, assoc ty item bounds and
// on type params defined by the closest item.

struct S1<T>(T) where (T): ?Sized; //~ ERROR this relaxed bound is not permitted here

struct S2<T>(T) where u8: ?Sized; //~ ERROR this relaxed bound is not permitted here

struct S3<T>(T) where &'static T: ?Sized; //~ ERROR this relaxed bound is not permitted here

trait Trait<'a> {}

struct S4<T>(T) where for<'a> T: ?Trait<'a>;
//~^ ERROR this relaxed bound is not permitted here
//~| ERROR bound modifier `?` can only be applied to `Sized`

struct S5<T>(*const T) where T: ?Trait<'static> + ?Sized;
//~^ ERROR bound modifier `?` can only be applied to `Sized`

impl<T> S1<T> {
    fn f() where T: ?Sized {} //~ ERROR this relaxed bound is not permitted here
}

// Test associated type bounds (ATB).
// issue: <https://github.com/rust-lang/rust/issues/135229>
struct S6<T>(T) where T: Iterator<Item: ?Sized>; //~ ERROR this relaxed bound is not permitted here

trait Tr: ?Sized {} //~ ERROR relaxed bounds are not permitted in supertrait bounds

// Test that relaxed `Sized` bounds are rejected in trait object types:

type O1 = dyn Tr + ?Sized; //~ ERROR relaxed bounds are not permitted in trait object types
type O2 = dyn ?Sized + ?Sized + Tr;
//~^ ERROR relaxed bounds are not permitted in trait object types
//~| ERROR relaxed bounds are not permitted in trait object types

fn main() {}
