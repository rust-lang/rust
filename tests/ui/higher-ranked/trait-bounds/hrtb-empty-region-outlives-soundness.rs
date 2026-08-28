// Regression test for https://github.com/rust-lang/rust/issues/161864

use std::fmt::Display;

trait ToObj<'a> {
    fn to_obj(self) -> Box<dyn Display + 'a>;
}

impl<'a, T: Display + 'a> ToObj<'a> for T {
    fn to_obj(self) -> Box<dyn Display + 'a> {
        Box::new(self)
    }
}

trait Super {
    fn yeet(&self);
}

impl<'a> Super for ()
where
    for<'b> &'b str: ToObj<'a>,
{
    fn yeet(&self) {
        let dangling = String::from("hello").as_str().to_obj();
        println!("{dangling}");
    }
}

trait Trait: Super {}
impl Trait for () {} //~ ERROR the type `&'b str` does not fulfill the required lifetime

fn main() {
    let _ = (&() as &dyn Trait).yeet();
}
