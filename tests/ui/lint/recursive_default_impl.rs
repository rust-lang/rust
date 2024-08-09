//@check-fail
struct Foo{
    bar: u32,
}

impl Default for Foo {
    fn default() -> Self {
    //~^ ERROR: recursive default impl
    //~| NOTE: will result in infinite recursion
    //~| HELP: ..default() in the Default impl does not apply a default for each struct field
    //~| NOTE: `#[deny(recursive_default_impl)]` on by default
        Self {
            ..Default::default()
            //~^ NOTE: recursive call site

        }
    }
}

fn main() {
    let _ = Foo::default();
}
