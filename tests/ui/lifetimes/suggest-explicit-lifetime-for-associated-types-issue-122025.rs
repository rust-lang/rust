struct MyType;

fn foo<F>()
where
    F: Iterator<Item = &MyType>, //~ `&` without an explicit lifetime name cannot be used here
{
}

fn bar() {
    fn baz<F>()
    where
        F: Iterator<Item = &MyType>, //~ `&` without an explicit lifetime name cannot be used here
    {
    }
}

fn function<T>()
where
    T: Iterator<Item = &MyType>, //~ `&` without an explicit lifetime name cannot be used here
{
    fn lambda<A>()
    where
        A: Iterator<Item = &u8>, //~ `&` without an explicit lifetime name cannot be used here
    {
    }
}

fn main() {}
