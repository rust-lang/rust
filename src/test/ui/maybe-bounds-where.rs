struct S1<T>(T) where (T): ?Sized;
//~^ ERROR `?Trait` bounds are only permitted at the point where a type parameter is declared

struct S2<T>(T) where u8: ?Sized;
//~^ ERROR `?Trait` bounds are only permitted at the point where a type parameter is declared

struct S3<T>(T) where &'static T: ?Sized;
//~^ ERROR `?Trait` bounds are only permitted at the point where a type parameter is declared

trait Trait<'a> {}

struct S4<T>(T) where for<'a> T: ?Trait<'a>;
//~^ ERROR `?Trait` bounds are only permitted at the point where a type parameter is declared

struct S5<T>(*const T) where T: ?Trait<'static> + ?Sized;
//~^ ERROR type parameter has more than one relaxed default bound
//~| WARN default bound relaxed for a type parameter

impl<T> S1<T> {
    fn f() where T: ?Sized {}
    //~^ ERROR `?Trait` bounds are only permitted at the point where a type parameter is declared
}

fn main() {
    let u = vec![1, 2, 3];
    let _s: S5<[u8]> = S5(&u[..]); // OK
}
