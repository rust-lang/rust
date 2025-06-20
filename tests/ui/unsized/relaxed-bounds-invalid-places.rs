// Test that relaxed bounds can only be placed on type parameters defined by the closest item
// (ignoring relaxed bounds inside `impl Trait` and in associated type defs here).

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

fn main() {
    let u = vec![1, 2, 3];
    let _s: S5<[u8]> = S5(&u[..]); // OK
}
