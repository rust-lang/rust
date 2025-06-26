pub enum Foo<T>
where:
//~^ ERROR expected one of `#`, `{`, lifetime, or type, found `:`
    T: Missing, {}

fn main() {}
