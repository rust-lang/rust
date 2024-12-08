trait Trait;
//~^ ERROR expected `{}`, found `;`

impl Trait for ();
//~^ ERROR expected `{}`, found `;`

enum Enum;
//~^ ERROR expected `{}`, found `;`

fn main() {}
