#[deprcated]    //~ ERROR E0658
fn foo() {}     //~| HELP a built-in attribute with a similar name exists
                //~| SUGGESTION deprecated
                //~| HELP add #![feature(custom_attribute)] to the crate attributes to enable

#[tests]        //~ ERROR E0658
fn bar() {}     //~| HELP a built-in attribute with a similar name exists
                //~| SUGGESTION test
                //~| HELP add #![feature(custom_attribute)] to the crate attributes to enable

#[rustc_err]    //~ ERROR E0658
fn main() {}    //~| HELP add #![feature(rustc_attrs)] to the crate attributes to enable
                // don't suggest rustc attributes
