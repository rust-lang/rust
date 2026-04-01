// ICE #147325: When the user mistakenly uses struct syntax to construct an enum,
// the field_idents and field_defaults functions will trigger an error

mod m {
    struct Priv1;
}

fn main() {
    Option { field1: m::Priv1 } //~ ERROR expected struct, variant or union type, found enum
    //~^ ERROR unit struct `Priv1` is private
}
