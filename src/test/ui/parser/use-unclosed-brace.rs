use foo::{bar, baz;
//~^ ERROR expected one of `,`, `::`, `as`, or `}`, found `;`

use std::fmt::Display;
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `std`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `;`

mod bar { }
//~^ ERROR expected identifier, found keyword `mod`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `bar`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `{`

mod baz { }
//~^ ERROR expected identifier, found keyword `mod`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `{`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `baz`
//~| ERROR expected one of `,` or `}`, found keyword `mod`

fn main() {}
//~^ ERROR expected identifier, found keyword `fn`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `(`
//~| ERROR expected one of `,` or `}`, found keyword `fn`
//~| ERROR expected one of `,`, `::`, `as`, or `}`, found `main`

//~ ERROR this file contains an unclosed delimiter