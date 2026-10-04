fn main() {}

fn foo() {
    #[derive(Copy, Clone)]
    struct Foo([u8; S]);
    //~^ ERROR cannot find value `S`
    //~| ERROR cannot find value `S`

    type U = impl Copy;
    //~^ ERROR `impl Trait` in type aliases is unstable
    let foo: U = Foo(());
    let Foo(()) = foo;
}
