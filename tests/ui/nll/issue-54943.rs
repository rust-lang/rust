fn foo<T: 'static>() { }

fn boo<'a>() {
    return;

    let x = foo::<&'a u32>();
    //~^ ERROR
}

fn main() {}
