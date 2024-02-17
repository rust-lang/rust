//@ edition: 2021
use u8 as imported_u8;
use unresolved as u8;
//~^ ERROR unresolved import `unresolved`

fn main() {}
