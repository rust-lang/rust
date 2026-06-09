struct Wrapper<T>(T);

fn foo<T>(foo: Wrapper<T>)
//~^ ERROR the size for values of type `T` cannot be known at compilation time
where
    T
    :
    ?
    Sized
{
    //
}

fn bar<T>(foo: Wrapper<T>)
//~^ ERROR the size for values of type `T` cannot be known at compilation time
where T: ?Sized
{
    //
}

fn qux<T>(foo: Wrapper<T>)
//~^ ERROR the size for values of type `T` cannot be known at compilation time
where
    T: ?Sized
{
    //
}


fn main() {}
