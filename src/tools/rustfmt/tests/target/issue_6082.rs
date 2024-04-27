macro_rules! test {
    ($T:ident, $b:lifetime) => {
        Box<$T<$b>>
    };
}
