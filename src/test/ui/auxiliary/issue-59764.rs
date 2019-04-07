pub mod foo {
    #[macro_export]
    macro_rules! makro {
        ($foo:ident) => {
            fn $foo() { }
        }
    }
}
