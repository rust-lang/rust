// rustfmt-error_on_line_overflow: true

#[rustfmt::skip] use one::two::three::four::five::six::seven::eight::night::ten::eleven::twelve::thirteen::fourteen::fiveteen;
#[rustfmt::skip]

use one::two::three::four::five::six::seven::eight::night::ten::eleven::twelve::thirteen::fourteen::fiveteen;

macro_rules! test_macro {
    ($($id:ident),*) => {};
}

macro_rules! test_macro2 {
    ($($id:ident),*) => {
        1
    };
}

fn main() {
    #[rustfmt::skip] test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
    #[rustfmt::skip]
    
    test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
}

fn test_local() {
    #[rustfmt::skip] let x = test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
    #[rustfmt::skip]
    
    let x = test_macro! { one, two, three, four, five, six, seven, eight, night, ten, eleven, twelve, thirteen, fourteen, fiveteen };
}

fn test_expr(_: [u32]) -> u32 {
    #[rustfmt::skip] test_expr([9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999]);
    #[rustfmt::skip]
    
    test_expr([9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999, 9999999999999])
}

#[rustfmt::skip] mod test { use one::two::three::four::five::six::seven::eight::night::ten::eleven::twelve::thirteen::fourteen::fiveteen; }
#[rustfmt::skip]

mod test { use one::two::three::four::five::six::seven::eight::night::ten::eleven::twelve::thirteen::fourteen::fiveteen; }
