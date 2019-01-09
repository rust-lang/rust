extern "C" fn foo(x: u8, ...);
//~^ ERROR only foreign functions are allowed to be variadic
//~| ERROR expected one of `->`, `where`, or `{`, found `;`
