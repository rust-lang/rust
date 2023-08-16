// Regression test for the ICE described in issue #86053.
//@error-in-other-file:unexpected `self` parameter in function
//@error-in-other-file:`...` must be the last argument of a C-variadic function
//@error-in-other-file:cannot find type `F` in this scope


#![feature(c_variadic)]
#![crate_type="lib"]

fn ordering4 < 'a , 'b     > ( a :            ,   self , self ,   self ,
    self , ... ,   self ,   self , ... ) where F : FnOnce ( & 'a & 'b usize ) {
}
