macro_rules! use_self {
    (
        impl $ty:ident {
            fn func(&$this:ident) {
                [fields($($field:ident)*)]
            }
        }
    ) => (
        impl  $ty {
            fn func(&$this) {
                let $ty { $($field),* } = $this;
            }
        }
    )
}
