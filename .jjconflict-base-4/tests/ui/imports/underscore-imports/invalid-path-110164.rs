//@ revisions: ed2015 ed2021
//@[ed2015] edition: 2015
//@[ed2021] edition: 2021
use self::*;
//~^ ERROR unresolved import `self::*`
use crate::*;
//~^ ERROR unresolved import `crate::*`
use _::a;
//~^ ERROR expected identifier, found reserved identifier `_`
use _::*;
//~^ ERROR expected identifier, found reserved identifier `_`

fn main() {
    use _::a;
    //~^ ERROR expected identifier, found reserved identifier `_`
    use _::*;
    //~^ ERROR expected identifier, found reserved identifier `_`
}
