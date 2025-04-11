struct Foo<'a,'b> {
    x: &'a isize,
    y: &'b isize,
}

impl<'a,'b> Foo<'a,'b> {
    fn bar(self:
           Foo<'b,'a>
    //~^ ERROR mismatched `self` parameter type
    //~| NOTE_NONVIRAL expected struct `Foo<'a, 'b>`
    //~| NOTE_NONVIRAL found struct `Foo<'b, 'a>`
    //~| NOTE_NONVIRAL lifetime mismatch
    //~| ERROR mismatched `self` parameter type
    //~| NOTE_NONVIRAL expected struct `Foo<'a, 'b>`
    //~| NOTE_NONVIRAL found struct `Foo<'b, 'a>`
    //~| NOTE_NONVIRAL lifetime mismatch
           ) {}
}

fn main() {}
