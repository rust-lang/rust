fn main() {}

fn foo() {
    #[derive(Copy, Clone)]
    struct Foo(NonExistent);
    //~^ ERROR cannot find type
    //~| ERROR cannot find type

    type U = impl Copy;
    //~^ ERROR `impl Trait` in type aliases is unstable
    let foo: U = Foo(());
    let Foo(()) = foo;
}
