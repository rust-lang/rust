// edition:2018
// compile-flags: -Zsave-analysis
// ~^ Also regression test for #69588

#![feature(register_attr)]
#![feature(register_tool)]

#![register_attr(attr)]
#![register_tool(tool)]

use attr as renamed_attr; // OK
use tool as renamed_tool; // OK

#[renamed_attr] //~ ERROR cannot use an explicitly registered attribute through an import
#[renamed_tool::attr] //~ ERROR cannot use a tool module through an import
                      //~| ERROR cannot use a tool module through an import
fn main() {}
