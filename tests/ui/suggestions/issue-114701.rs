enum Enum<T> { , SVariant { v: T }, UVariant } //~ ERROR expected identifier, found `,`

macro_rules! is_variant {
    (TSVariant, ) => (!);
    (SVariant, ) => (!);
    (UVariant, $expr:expr) => (is_variant!(@check UVariant, {}, $expr));
    (@check $variant:ident, $matcher:tt, $expr:expr) => (
        assert!(if let Enum::$variant::<()> $matcher = $expr () else { false }, //~ ERROR this `if` expression
                );
    );
}

fn main() {
    is_variant!(UVariant, Enum::<()>::UVariant); //~ ERROR expected function
}
