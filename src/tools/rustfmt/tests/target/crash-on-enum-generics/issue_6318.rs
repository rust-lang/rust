// rustfmt-max_width: 80
fn my_fn() {
    enum MyEnum
    where
    SomeTypeA___: SomeTrait__<
            _A,
            Archived = <SomeTypeB____ as SomeTrait__<
                Option<[u8; 4]>,
            >>::Archived,
        >,
    // left unformatted since formatting where clause fails 
    {
    }
}
