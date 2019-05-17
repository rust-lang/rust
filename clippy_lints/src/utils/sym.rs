#[macro_export]
macro_rules! sym {
    ($tt:tt) => {
        syntax::symbol::Symbol::intern(stringify!($tt))
    };
}
