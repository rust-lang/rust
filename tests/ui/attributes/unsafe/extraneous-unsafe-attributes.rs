//@ edition: 2024

#[unsafe(cfg(false))] //~ ERROR: is not an unsafe attribute
fn a() {}

#[unsafe(cfg_attr(false, allow(dead_code)))] //~ ERROR: is not an unsafe attribute
fn b() {}

#[unsafe(test)] //~ ERROR: is not an unsafe attribute
fn aa() {}

#[unsafe(ignore = "test")]
//~^ ERROR: is not an unsafe attribute
//~| ERROR `#[ignore]` should only be applied to functions annotated with `#[test]` or `#[bench]`
fn bb() {}

#[unsafe(should_panic(expected = "test"))]
//~^ ERROR: is not an unsafe attribute
//~| ERROR `#[should_panic]` should only be applied to functions annotated with `#[test]` or `#[bench]`
fn cc() {}

#[unsafe(macro_use)] //~ ERROR: is not an unsafe attribute
mod inner {
    #[unsafe(macro_export)] //~ ERROR: is not an unsafe attribute
    macro_rules! m {
        () => {};
    }
}

#[unsafe(used)] //~ ERROR: is not an unsafe attribute
static FOO: usize = 0;

fn main() {
    let _a = cfg!(unsafe(foo));
    //~^ ERROR: expected identifier, found keyword `unsafe`
    //~^^ ERROR: invalid predicate `r#unsafe`
}
