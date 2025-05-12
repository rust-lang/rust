#[macro_export]
macro_rules! make_item (
    ($name: ident) => (pub const $name: usize = 42;)
);
