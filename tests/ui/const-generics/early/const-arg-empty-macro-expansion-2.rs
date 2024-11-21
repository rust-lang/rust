impl<T>
    Foo<
        //~^ ERROR: cannot find type `Foo` in this scope
        T,
        {
            thread_local! { pub static FOO : Foo = Foo { } ; }
            //~^ ERROR: cannot find type `Foo` in this scope
            //~| ERROR: cannot find type `Foo` in this scope
            //~| ERROR: cannot find type `Foo` in this scope
            //~| ERROR: cannot find type `Foo` in this scope
            //~| ERROR: cannot find type `Foo` in this scope
            //~| ERROR: cannot find struct, variant or union type `Foo` in this scope
        },
    >
{
}

fn main() {}
