macro_rules! test_macro {
    ( $( $t:ty ),* $(),*) => {
        enum SomeEnum {
            $( $t, )* //~ ERROR expected identifier, found invisible open delimiter
        };
    };
}

fn main() {
    test_macro!(String,);
}
