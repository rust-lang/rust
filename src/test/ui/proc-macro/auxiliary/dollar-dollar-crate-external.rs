#![feature(macro_metavar_expr)]

#[macro_export]
macro_rules! define_macro {
    ($m:ident => $item:ident) => {
        macro_rules! $m {
            () => {
                $$crate::$item
            };
        }
    };
}
