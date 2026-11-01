struct Foo<'a> {
    x: Option<&'a u32>,
}

fn main() {
    let mut x = Foo { x: None };
    let y = 0;
    x.x = Some(&y);
    //~^ ERROR `y` does not live long enough [E0597]
}

impl<'a> Drop for Foo<'a> { fn drop(&mut self) { } }
