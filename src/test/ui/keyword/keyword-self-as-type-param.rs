// Regression test of #36638.

struct Foo<Self>(Self);
//~^ ERROR expected identifier, found keyword `Self`
//~^^ ERROR E0392

trait Bar<Self> {}
//~^ ERROR expected identifier, found keyword `Self`

fn main() {}
