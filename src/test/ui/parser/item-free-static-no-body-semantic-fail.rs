// Semantically, a free `static` item cannot omit its body.

fn main() {}

static A: u8; //~ ERROR free static item without body
static B; //~ ERROR free static item without body
//~^ ERROR missing type for `static` item

static mut C: u8; //~ ERROR free static item without body
static mut D; //~ ERROR free static item without body
//~^ ERROR missing type for `static mut` item
