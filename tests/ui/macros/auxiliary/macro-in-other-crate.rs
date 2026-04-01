#[macro_export]
macro_rules! mac {
    ($ident:ident) => { let $ident = 42; }
}

#[macro_export]
macro_rules! inline {
    () => ()
}

#[macro_export]
macro_rules! from_prelude {
    () => ()
}
