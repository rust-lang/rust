trait Foo<T> {
    const BAR: bool
        where //~ ERROR: expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `=`, found keyword `where`
            Self: Sized;
}

trait Cake {}
impl Cake for () {}

fn foo(_: &dyn Foo<()>) {}
fn bar(_: &dyn Foo<i32>) {}

fn main() {}
