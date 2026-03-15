// #141081
fn bad_lifetime_name<'🐛🐛🐛family👨‍👩‍👧‍👦>(_: &'🐛🐛🐛family👨‍👩‍👧‍👦 ()) {}
//~^ ERROR: lifetimes cannot contain emoji
//~| ERROR: lifetimes cannot contain emoji
fn main() {
    '🐛: { //~ ERROR: lifetimes cannot contain emoji
        todo!();
    };
}
