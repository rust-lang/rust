const FOO: impl Copy = 42;
//~^ ERROR `impl Trait` not allowed

static BAR: impl Copy = 42;
//~^ ERROR `impl Trait` not allowed

fn main() {
    let foo = impl Copy = 42;
//~^ ERROR expected expression, found keyword `impl`
    let foo: impl Copy = 42;
}
