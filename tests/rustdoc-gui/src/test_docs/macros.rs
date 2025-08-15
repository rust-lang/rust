#[macro_export]
macro_rules! a{ () => {}}
#[macro_export]
macro_rules! b{ () => {}}

// An attribute bang macro.
#[macro_export]
macro_rules! attr_macro {
    attr() () => {};
    () => {};
}
