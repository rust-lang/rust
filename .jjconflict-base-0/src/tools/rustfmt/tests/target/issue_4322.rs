trait Bar {
    type X<'a>
    where
        Self: 'a;
}
