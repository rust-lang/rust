#[macro_export]
macro_rules! reemit_legacy {
    ($($tok:tt)*) => ($($tok)*)
}

#[macro_export]
macro_rules! say_hello_extern {
    ($macname:ident) => ( $macname! { "Hello, world!" })
}
