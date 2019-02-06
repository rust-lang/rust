// rustfmt-error_on_line_overflow: true

macro_rules! test_macro {
    ($($id:ident),*) => {};
}

fn main() {
    #[rustfmt::skip] test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
    #[rustfmt::skip]
    
    test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
}
