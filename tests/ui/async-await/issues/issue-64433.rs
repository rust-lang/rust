// Regression test for issue #64433.
//
// See issue-64391-2.rs for more details, as that was fixed by the
// same PR.
//
//@ check-pass
//@ edition:2018

#[derive(Debug)]
struct A<'a> {
    inner: Vec<&'a str>,
}

struct B {}

impl B {
    async fn something_with_a(&mut self, a: A<'_>) -> Result<(), String> {
        println!("{:?}", a);
        Ok(())
    }
}

async fn can_error(some_string: &str) -> Result<(), String> {
    let a = A { inner: vec![some_string, "foo"] };
    let mut b = B {};
    Ok(b.something_with_a(a).await.map(drop)?)
}

fn main() {
}
