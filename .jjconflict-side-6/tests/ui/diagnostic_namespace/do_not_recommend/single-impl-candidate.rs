//@ reference: attributes.diagnostic.do_not_recommend.intro

pub struct Internal;

#[diagnostic::do_not_recommend]
impl From<Internal> for &str {
    fn from(_: Internal) -> Self {
        ""
    }
}

fn foo<'a, T>(_t: T)
where
    T: Into<&'a str>,
{
}

fn main() {
    foo(String::new());
    //~^ ERROR the trait bound `&str: From<String>` is not satisfied
}
