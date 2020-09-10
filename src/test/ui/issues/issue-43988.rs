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
    //~^^ ERROR attribute should be applied to a struct, enum, or union

    #[repr(something_not_real)]
    loop {
        ()
    };
    //~^^^^ ERROR attribute should be applied to a struct, enum, or union

    #[repr]
    let _y = "123";
    //~^^ ERROR malformed `repr` attribute

    fn foo() {}

    #[inline(ABC)]
    foo();
    //~^^ ERROR attribute should be applied to function or closure

    let _z = #[repr] 1;
    //~^ ERROR malformed `repr` attribute
}
