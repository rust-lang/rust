// build-pass (FIXME(62277): could be check-pass?)

trait Foo<'a> {
    const C: &'a u32;
}

impl<'a, T> Foo<'a> for T {
    const C: &'a u32 = &22;
}

fn foo() {
    let a = 22;
    match &a {
        <() as Foo<'static>>::C => { }
        &_ => { }
    }
}

fn main() {}
