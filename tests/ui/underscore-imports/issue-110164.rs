use self::*;
//~^ ERROR unresolved import `self::*`
use crate::*;
//~^ ERROR unresolved import `crate::*`
use _::a;
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR unresolved import `_`
use _::*;
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR unresolved import `_`

fn main() {
    use _::a;
    //~^ ERROR expected identifier, found reserved identifier `_`
    //~| ERROR unresolved import `_`
    use _::*;
    //~^ ERROR expected identifier, found reserved identifier `_`
    //~| ERROR unresolved import `_`
}
