struct Foo<'a> {
    data: &'a[u8],
}

impl <'a> Foo<'a>{
    fn bar(self: &mut Foo) {
    //~^ ERROR mismatched `self` parameter type
    //~| NOTE_NONVIRAL expected struct `Foo<'a>`
    //~| NOTE_NONVIRAL found struct `Foo<'_>`
    //~| NOTE_NONVIRAL lifetime mismatch
    //~| ERROR mismatched `self` parameter type
    //~| NOTE_NONVIRAL expected struct `Foo<'a>`
    //~| NOTE_NONVIRAL found struct `Foo<'_>`
    //~| NOTE_NONVIRAL lifetime mismatch
    }
}

fn main() {}
