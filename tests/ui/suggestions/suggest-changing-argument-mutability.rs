fn takes_mut(_: &mut String) {}

struct S;
impl S {
    fn mutate(&mut self, _: &mut i32) {}
}

fn main() {
    let mut s = String::new();
    takes_mut(&s);
    //~^ ERROR mismatched types

    let mut val = 42;
    S.mutate(&val);
    //~^ ERROR mismatched types
}
