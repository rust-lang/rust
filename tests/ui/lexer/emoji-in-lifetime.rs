// #141081
fn bad_lifetime_name<
    '🐛🐛🐛family👨‍👩‍👧‍👦,//~ ERROR: identifiers cannot contain emoji
    '12, //~ ERROR: lifetimes cannot start with a number
    'a🐛, //~ ERROR: identifiers cannot contain emoji
    '1🐛, //~ ERROR: identifiers cannot contain emoji
    //~^ ERROR: lifetimes cannot start with a number
    '1, //~ ERROR: lifetimes cannot start with a number
    'a‌b // bare zero-width-joiners are accepted as XID_Continue
>() {}

fn main() {
    'a🐛: { // pointed at on the error from line 5
        todo!();
    };
}
