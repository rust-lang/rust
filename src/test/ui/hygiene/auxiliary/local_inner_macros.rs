#[macro_export]
macro_rules! helper1 {
    () => ( struct S; )
}

#[macro_export(local_inner_macros)]
macro_rules! helper2 {
    () => ( helper1!(); )
}

#[macro_export(local_inner_macros)]
macro_rules! public_macro {
    () => ( helper2!(); )
}

#[macro_export(local_inner_macros)]
macro_rules! public_macro_dynamic {
    ($helper: ident) => ( $helper!(); )
}
