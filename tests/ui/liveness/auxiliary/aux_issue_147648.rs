#[macro_export]
macro_rules! unused_assign {
    ($x:ident) => {
        let mut $x = 1;
        $x = 2;
    };
}
