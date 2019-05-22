#![feature(stmt_expr_attributes)]

fn main() {

    #[inline]
    let _a = 4;
    //~^^ ERROR attribute should be applied to function or closure


    #[inline(XYZ)]
    let _b = 4;
    //~^^ ERROR attribute should be applied to function or closure

    #[repr(nothing)]
    let _x = 0;
    //~^^ ERROR attribute should not be applied to a statement

    #[repr(something_not_real)]
    loop {
        ()
    };
    //~^^^^ ERROR attribute should not be applied to an expression

    #[repr]
    let _y = "123";
    //~^^ ERROR attribute should not be applied to a statement
    //~| ERROR malformed `repr` attribute

    fn foo() {}

    #[inline(ABC)]
    foo();
    //~^^ ERROR attribute should be applied to function or closure

    let _z = #[repr] 1;
    //~^ ERROR attribute should not be applied to an expression
    //~| ERROR malformed `repr` attribute
}
