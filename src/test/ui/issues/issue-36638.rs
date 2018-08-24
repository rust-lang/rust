// compile-flags: -Z continue-parse-after-error

struct Foo<Self>(Self);
//~^ ERROR expected identifier, found keyword `Self`

trait Bar<Self> {}
//~^ ERROR expected identifier, found keyword `Self`

fn main() {}
