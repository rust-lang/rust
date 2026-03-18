// #141081
fn bad_lifetime_name<
    '🐛🐛🐛family👨‍👩‍👧‍👦,//~ ERROR: lifetimes cannot have emoji
    '12, //~ ERROR: lifetimes cannot start with a number
    'a🐛, //~ ERROR: lifetimes cannot have emoji
    '1🐛, //~ ERROR: invalid lifetime name
    '1, //~ ERROR: lifetimes cannot start with a number
    'a‌b // bare zero-width-joiners are accepted as XID_Continue
>() {}






fn main() {
    '🐛: { //~ ERROR: lifetimes cannot have emoji
        todo!();
    };
}
