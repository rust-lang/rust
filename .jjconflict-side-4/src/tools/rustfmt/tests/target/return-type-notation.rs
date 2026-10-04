fn rtn()
where
    T: Trait<method(..): Send + 'static>,
    T::method(..): Send + 'static,
{
}

fn test() {
    let x: T::method(..);
}
