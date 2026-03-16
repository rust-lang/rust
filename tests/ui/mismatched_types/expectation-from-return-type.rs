use std::ptr;

fn main() { //~ NOTE: this implicit `()` return type influences the call expression's return type
    let a = 0;
    ptr::read(&a)
    //~^ ERROR: mismatched types
    //~| NOTE: expected `*const ()`, found `&{integer}`
    //~| NOTE: arguments to this function are incorrect
    //~| NOTE: expected raw pointer
    //~| NOTE: function defined here
}

fn foo() { //~ NOTE: this implicit `()` return type influences the call expression's return type
    let a = 0;
    return ptr::read(&a);
    //~^ ERROR: mismatched types
    //~| NOTE: expected `*const ()`, found `&{integer}`
    //~| NOTE: arguments to this function are incorrect
    //~| NOTE: expected raw pointer
    //~| NOTE: function defined here
}
