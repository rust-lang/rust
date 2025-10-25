#[macro_export]
macro_rules! a{ () => {}}
#[macro_export]
macro_rules! b{ () => {}}
/// An attribute bang macro.
#[macro_export]
macro_rules! attr_macro {
    attr() () => {};
    () => {};
}

/// An attribute bang macro.
#[macro_export]
macro_rules! derive_macro {
    derive() () => {};
    () => {};
}

#[macro_export]
macro_rules! one_for_all_macro {
    attr() () => {};
    derive() () => {};
    () => {};
}
