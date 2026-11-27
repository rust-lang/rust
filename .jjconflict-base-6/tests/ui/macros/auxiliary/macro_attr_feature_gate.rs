#![feature(macro_attr)]

#[macro_export]
macro_rules! identity {
    attr() { $item:item } => {
        $item
    };
}
