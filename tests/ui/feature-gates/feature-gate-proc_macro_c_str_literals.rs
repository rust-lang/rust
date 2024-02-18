//@ edition: 2021
//@ force-host
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::Literal;

fn test() {
    Literal::c_string(c"a"); //~ ERROR use of unstable library feature 'proc_macro_c_str_literals'
}
