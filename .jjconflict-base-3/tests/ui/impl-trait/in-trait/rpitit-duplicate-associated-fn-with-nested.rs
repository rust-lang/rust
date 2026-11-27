// issue#143560

trait T {
    type Target;
}

trait Foo {
    fn foo() -> impl T<Target = impl T<Target = impl Sized>>;
    fn foo() -> impl Sized;
    //~^ ERROR: the name `foo` is defined multiple times
}

trait Bar {
    fn foo() -> impl T<Target = impl T<Target = impl Sized>>;
    fn foo() -> impl T<Target = impl T<Target = impl Sized>>;
    //~^ ERROR: the name `foo` is defined multiple times
}

struct S<T> {
    a: T
}

trait Baz {
    fn foo() -> S<impl T<Target = S<S<impl Sized>>>>;
    fn foo() -> S<impl T<Target = S<S<impl Sized>>>>;
    //~^ ERROR: the name `foo` is defined multiple times
}

struct S1<T1, T2> {
    a: T1,
    b: T2
}

trait Qux {
    fn foo() -> S1<
        impl T<Target = impl T<Target = impl Sized>>,
        impl T<Target = impl T<Target = S<impl Sized>>>
        >;
    fn foo() -> S1<
        impl T<Target = impl T<Target = impl Sized>>,
        impl T<Target = impl T<Target = S<impl Sized>>>
        >;
    //~^^^^ ERROR: the name `foo` is defined multiple times
}

trait T0<T> {
    type Target;
}
trait T1<T> {}

trait X {
    fn a() -> impl T0<(), Target = impl T1<()>>;
    fn a() -> impl T0<(), Target = impl T1<()>>;
    //~^ ERROR the name `a` is defined multiple times
    fn a() -> impl T0<(), Target = impl T1<()>>;
    //~^ ERROR the name `a` is defined multiple times
}

fn main() {}
