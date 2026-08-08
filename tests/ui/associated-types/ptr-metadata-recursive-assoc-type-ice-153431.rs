// Regression test for <https://github.com/rust-lang/rust/issues/153431>.
// Used to ICE in `ptr_metadata_ty_or_tail` when computing pointer metadata
// for a struct whose tail field has a recursive associated type.
//~^^^ ERROR overflow evaluating the requirement `<Bar as Trait>::Assoc2 == _`

trait Trait {
    type Assoc2;
}

struct Bar;
impl Trait for Bar {
    type Assoc2 = Result<(), <Bar as Trait>::Assoc2>;
    //~^ ERROR overflow evaluating the requirement `<Bar as Trait>::Assoc2 == _`
}

struct Foo {
    field: <Bar as Trait>::Assoc2,
    //~^ ERROR overflow evaluating the requirement `<Bar as Trait>::Assoc2 == _`
}

static BAR: u8 = 42;
static FOO2: &Foo = unsafe { std::mem::transmute(&BAR) };

fn main() {}
