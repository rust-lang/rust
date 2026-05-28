// Auxiliary lib for the issue 147973 regression test with ICEs due to overlapping spans.

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
