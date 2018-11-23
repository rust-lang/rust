fn foo<T: 'static>() { }

fn main<'a>() {
    return;

    let x = foo::<&'a u32>();
}
