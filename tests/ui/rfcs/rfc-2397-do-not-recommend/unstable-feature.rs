trait Foo {
}

#[do_not_recommend]
//~^ ERROR the `#[do_not_recommend]` attribute is an experimental feature
impl Foo for i32 {
}

fn main() {
}
