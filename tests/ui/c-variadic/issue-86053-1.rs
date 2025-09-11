// Regression test for the ICE described in issue #86053.

#![feature(c_variadic)]
#![crate_type="lib"]

fn ordering4 < 'a , 'b     > ( a :            ,   self , self ,   self ,
    //~^ ERROR expected type, found `,`
    //~| ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    self , ... ,   self ,   self , ... ) where F : FnOnce ( & 'a & 'b usize ) {
    //~^ ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    //~| ERROR `...` must be the last argument of a C-variadic function
    //~| ERROR `...` is not supported for non-extern functions
    //~| ERROR cannot find type `F` in this scope
}
