#![feature(coroutines, stmt_expr_attributes)]
#![feature(auto_traits)]
#![feature(negative_impls)]

auto trait Foo {}

struct No;

impl !Foo for No {}

struct A<'a, 'b>(&'a mut bool, &'b mut bool, No);

impl<'a, 'b: 'a> Foo for A<'a, 'b> {}

struct OnlyFooIfStaticRef(No);
impl Foo for &'static OnlyFooIfStaticRef {}

struct OnlyFooIfRef(No);
impl<'a> Foo for &'a OnlyFooIfRef {}

fn assert_foo<T: Foo>(f: T) {}

fn main() {
    // Make sure 'static is erased for coroutine interiors so we can't match it in trait selection
    let x: &'static _ = &OnlyFooIfStaticRef(No);
    let generator = #[coroutine] move || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(generator);
    //~^ ERROR implementation of `Foo` is not general enough

    // Allow impls which matches any lifetime
    let x = &OnlyFooIfRef(No);
    let generator = #[coroutine] move || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(generator); // ok

    // Disallow impls which relates lifetimes in the coroutine interior
    let generator = #[coroutine] move || {
        let a = A(&mut true, &mut true, No);
        //~^ ERROR borrow may still be in use when coroutine yields
        //~| ERROR borrow may still be in use when coroutine yields
        yield;
        assert_foo(a);
    };
    assert_foo(generator);
    //~^ ERROR not general enough
}
