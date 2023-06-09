// Regression test for the ICE described in issue #86053.
// error-pattern:unexpected `self` parameter in function
// error-pattern:`...` must be the last argument of a C-variadic function
// error-pattern:cannot find type `F` in this scope


#![feature(c_variadic)]
#![crate_type="lib"]

fn ordering4 < 'a , 'b     > ( a :            ,   self , self ,   self ,
    self , ... ,   self ,   self , ... ) where F : FnOnce ( & 'a & 'b usize ) {
}
