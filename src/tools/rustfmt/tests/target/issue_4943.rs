impl SomeStruct {
    fn process<T>(v: T) -> <Self as GAT>::R<T>
    where
        Self: GAT<R<T> = T>,
    {
        SomeStruct::do_something(v)
    }
}
