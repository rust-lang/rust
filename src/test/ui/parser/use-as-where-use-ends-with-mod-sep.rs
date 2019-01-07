// compile-flags: -Z continue-parse-after-error

use std::any:: as foo; //~ ERROR expected identifier, found keyword `as`
//~^ ERROR: expected one of `::`, `;`, or `as`, found `foo`
