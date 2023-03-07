// run-pass
#![allow(unused_parens)]
#![allow(non_camel_case_types)]


enum colour { red(isize, isize), green, }

impl PartialEq for colour {
    fn eq(&self, other: &colour) -> bool {
        match *self {
            colour::red(a0, b0) => {
                match (*other) {
                    colour::red(a1, b1) => a0 == a1 && b0 == b1,
                    colour::green => false,
                }
            }
            colour::green => {
                match (*other) {
                    colour::red(..) => false,
                    colour::green => true
                }
            }
        }
    }
    fn ne(&self, other: &colour) -> bool { !(*self).eq(other) }
}

fn f() { let x = colour::red(1, 2); let y = colour::green; assert!((x != y)); }

pub fn main() { f(); }
