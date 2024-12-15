trait Foo<T> {
    type Bar
    where
        Self: Foo<()>;
}

trait Cake {}
impl Cake for () {}

fn foo(_: &dyn Foo<()>) {} //~ ERROR: the value of the associated type `Bar` in `Foo<()>` must be specified
fn bar(_: &dyn Foo<i32>) {} //~ ERROR: the value of the associated type `Bar` in `Foo<i32>` must be specified

fn main() {}
