#![feature(stmt_expr_attributes)]

fn main() {

    #[inline]
    let _a = 4;
    //~^^ ERROR attribute cannot be used on


    #[inline(XYZ)]
    let _b = 4;
    //~^^ ERROR malformed `inline` attribute
    //~| ERROR attribute cannot be used on

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
    //~| ERROR attribute cannot be used on

    let _z = #[repr] 1;
    //~^ ERROR malformed `repr` attribute
}
