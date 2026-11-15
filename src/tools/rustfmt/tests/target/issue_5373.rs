fn foo()
where
    for<T> T: Clone,
{
}

fn bar<T>()
where
    T: for<U> Clone,
{
}
