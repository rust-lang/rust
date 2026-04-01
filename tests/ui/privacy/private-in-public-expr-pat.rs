// Patterns and expressions are not interface parts and don't produce private-in-public errors.

//@ build-pass (FIXME(62277): could be check-pass?)

struct Priv1(usize);
struct Priv2;

pub struct Pub(Priv2);

pub fn public_expr(_: [u8; Priv1(0).0]) {} // OK
pub fn public_pat(Pub(Priv2): Pub) {} // OK

fn main() {}
