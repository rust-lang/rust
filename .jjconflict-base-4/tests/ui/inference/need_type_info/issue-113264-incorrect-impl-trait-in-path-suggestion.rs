trait T {}

struct S {}

impl S {
    fn owo(&self, _: Option<&impl T>) {}
}

fn main() {
    (S {}).owo(None)
    //~^ ERROR type annotations needed
}
