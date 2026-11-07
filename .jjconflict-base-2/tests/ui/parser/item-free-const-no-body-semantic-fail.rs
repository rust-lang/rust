// Semantically, a free `const` item cannot omit its body.

fn main() {}

const A: u8; //~ ERROR free constant item without body
const B; //~ ERROR free constant item without body
//~^ ERROR missing type for `const` item
