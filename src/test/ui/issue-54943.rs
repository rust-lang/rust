fn foo<T: 'static>() { }

fn main<'a>() {
    return;

    let x = foo::<&'a u32>();
    //~^ ERROR the type `&'a u32` does not fulfill the required lifetime [E0477]
}
