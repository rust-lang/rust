fn main() {}

#[deprecated(
    since = "since" //~ ERROR attribute items not separated with `,`
    note = "note"
)]
fn f0() {}
