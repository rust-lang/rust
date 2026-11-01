#[macro_export]
macro_rules! identity {
    ($x:ident) => {
        $x
    };
}

#[macro_export]
macro_rules! do_loop {
    ($x:ident) => {
        for $crate::identity!($x) in $x {}
    };
}
