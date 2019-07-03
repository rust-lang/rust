use std::cell::Cell;

trait Foo<'a> {
    const C: Option<Cell<&'a u32>>;
}

impl<'a, T> Foo<'a> for T {
    const C: Option<Cell<&'a u32>> = None;
}

fn main() {
    let a = 22;
    let b = Some(Cell::new(&a));
    //~^ ERROR `a` does not live long enough [E0597]
    match b {
        <() as Foo<'static>>::C => { }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN will become a hard error in a future release
        _ => { }
    }
}
