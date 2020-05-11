// edition:2018

struct S<'a>(&'a i32);

impl<'a> S<'a> {
    async fn new(i: &'a i32) -> Self {
    //~^ ERROR `async fn` return type cannot contain a projection or `Self` that references lifetimes from a parent scope
        S(&22)
    }
}

fn main() {}
