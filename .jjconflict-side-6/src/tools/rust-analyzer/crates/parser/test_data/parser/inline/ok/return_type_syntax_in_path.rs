fn foo<T>()
where
    T::method(..): Send,
    method(..): Send,
    method::(..): Send,
{}
