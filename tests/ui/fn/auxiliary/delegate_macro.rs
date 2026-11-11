#[macro_export]
macro_rules! delegate {
    ($method:ident) => {
        <Self>::$method(8)
    };
}
