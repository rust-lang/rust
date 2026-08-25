#[macro_export]
macro_rules! other_macro {
    ($x:ident) => {{
        $x += 2;
    }}
}
