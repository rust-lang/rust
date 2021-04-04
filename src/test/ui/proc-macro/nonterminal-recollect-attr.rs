// check-pass
// aux-build:nonterminal-recollect-attr.rs

extern crate nonterminal_recollect_attr;
use nonterminal_recollect_attr::*;

macro_rules! my_macro {
    ($v:ident) => {
        #[first_attr]
        $v struct Foo {
            field: u8
        }
    }
}

my_macro!(pub);
fn main() {}
