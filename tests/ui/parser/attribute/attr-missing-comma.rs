fn main() {}

#[deprecated(
    since = "since" //~ ERROR attribute items not separated with `,`
    note = "note"
)]
fn f0() {}

#[link(
    name = "name" //~ ERROR attribute items not separated with `,`
    kind = "static"
    modifiers = "modifiers"
)]
fn f1() {}
