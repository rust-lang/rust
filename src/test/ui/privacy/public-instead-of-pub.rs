// Checks what happens when `public` is used instead of the correct, `pub`
// edition:2018
public struct MyStruct;
//~^ ERROR 3:8: 3:14: expected one of `!` or `::`, found keyword `struct`
