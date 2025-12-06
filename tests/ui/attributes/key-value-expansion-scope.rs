#![doc = in_root!()] //~ ERROR cannot find macro `in_root`
                     //~| WARN this was previously accepted by the compiler
#![doc = in_mod!()] //~ ERROR cannot find macro `in_mod` in this scope
                    //~| ERROR attribute value must be a literal
#![doc = in_mod_escape!()] //~ ERROR cannot find macro `in_mod_escape`
                           //~| WARN this was previously accepted by the compiler
#![doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                      //~| ERROR attribute value must be a literal

#[doc = in_root!()] //~ ERROR cannot find macro `in_root` in this scope
                    //~| ERROR attribute value must be a literal
#[doc = in_mod!()] //~ ERROR cannot find macro `in_mod` in this scope
                   //~| ERROR attribute value must be a literal
#[doc = in_mod_escape!()] //~ ERROR cannot find macro `in_mod_escape` in this scope
                          //~| ERROR attribute value must be a literal
#[doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                     //~| ERROR attribute value must be a literal
fn before() {
    #![doc = in_root!()] //~ ERROR cannot find macro `in_root` in this scope
                         //~| ERROR attribute value must be a literal
    #![doc = in_mod!()] //~ ERROR cannot find macro `in_mod` in this scope
                        //~| ERROR attribute value must be a literal
    #![doc = in_mod_escape!()] //~ ERROR cannot find macro `in_mod_escape` in this scope
                               //~| ERROR attribute value must be a literal
    #![doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                          //~| ERROR attribute value must be a literal
}

macro_rules! in_root { () => { "" } }

#[doc = in_mod!()] //~ ERROR cannot find macro `in_mod`
                   //~| WARN this was previously accepted by the compiler
mod macros_stay {
    #![doc = in_mod!()] //~ ERROR cannot find macro `in_mod`
                        //~| WARN this was previously accepted by the compiler

    macro_rules! in_mod { () => { "" } }

    #[doc = in_mod!()] // OK
    fn f() {
        #![doc = in_mod!()] // OK
    }
}

#[macro_use]
#[doc = in_mod_escape!()] //~ ERROR cannot find macro `in_mod_escape`
                          //~| WARN this was previously accepted by the compiler
mod macros_escape {
    #![doc = in_mod_escape!()] //~ ERROR cannot find macro `in_mod_escape`
                               //~| WARN this was previously accepted by the compiler

    macro_rules! in_mod_escape { () => { "" } }

    #[doc = in_mod_escape!()] // OK
    fn f() {
        #![doc = in_mod_escape!()] // OK
    }
}

#[doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                     //~| ERROR attribute value must be a literal
fn block() {
    #![doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                          //~| ERROR attribute value must be a literal

    macro_rules! in_block { () => { "" } }

    #[doc = in_block!()] // OK
    fn f() {
        #![doc = in_block!()] // OK
    }
}

#[doc = in_root!()] // OK
#[doc = in_mod!()] //~ ERROR cannot find macro `in_mod` in this scope
                   //~| ERROR attribute value must be a literal
#[doc = in_mod_escape!()] // OK
#[doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                     //~| ERROR attribute value must be a literal
fn after() {
    #![doc = in_root!()] // OK
    #![doc = in_mod!()] //~ ERROR cannot find macro `in_mod` in this scope
                        //~| ERROR attribute value must be a literal
    #![doc = in_mod_escape!()] // OK
    #![doc = in_block!()] //~ ERROR cannot find macro `in_block` in this scope
                          //~| ERROR attribute value must be a literal
}

fn main() {}
