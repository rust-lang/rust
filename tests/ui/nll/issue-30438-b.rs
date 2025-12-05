// Modified regression test for Issue #30438 that exposed an
// independent issue (see discussion on ticket).

use std::ops::Index;

struct Test<'a> {
    s: &'a String
}

impl <'a> Index<usize> for Test<'a> {
    type Output = Test<'a>;
    fn index(&self, _: usize) -> &Self::Output {
        &Test { s: &self.s}
        //~^ ERROR: cannot return reference to temporary value
    }
}

fn main() {
    let s = "Hello World".to_string();
    let test = Test{s: &s};
    let r = &test[0];
    println!("{}", test.s); // OK since test is valid
    println!("{}", r.s); // Segfault since value pointed by r has already been dropped
}
