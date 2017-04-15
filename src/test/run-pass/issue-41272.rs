struct Foo;

impl Foo {
    fn bar(&mut self) -> bool { true }
}

/* This causes E0301. By fixing issue #41272 this problem should vanish */
fn iflet_issue(foo: &mut Foo) {
    if let Some(_) = Some(true) {
    } else if foo.bar() {}
}

fn main() {}
