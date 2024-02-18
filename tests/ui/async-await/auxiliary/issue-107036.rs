//@ edition:2021

pub trait T {}
impl T for () {}

pub struct S {}

impl S {
    pub async fn f<'a>(&self) -> impl T + 'a {
        ()
    }
}
