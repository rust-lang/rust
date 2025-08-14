#![feature(stmt_expr_attributes)]

fn main() {

    #[inline]
    let _a = 4;
    //~^^ ERROR attribute should be applied to function or closure


    #[inline(XYZ)]
    let _b = 4;
    //~^^ ERROR malformed `inline` attribute

    #[repr(nothing)]
    let _x = 0;
    //~^^ ERROR E0552

    #[repr(something_not_real)]
    loop {
        ()
    };
    //~^^^^ ERROR E0552

    #[repr]
    let _y = "123";
    //~^^ ERROR malformed `repr` attribute

    fn foo() {}

    #[inline(ABC)]
    foo();
    //~^^ ERROR malformed `inline` attribute

    let _z = #[repr] 1;
    //~^ ERROR malformed `repr` attribute
}
