//@ check-pass
//@ edition:2018

struct S<'a>(&'a i32);

impl<'a> S<'a> {
    async fn new(i: &'a i32) -> Result<Self, ()> {
        Ok(S(&22))
    }
}

fn main() {}
