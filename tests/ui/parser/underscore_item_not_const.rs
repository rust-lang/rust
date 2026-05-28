// Test that various non-const items do not syntactically permit `_` as a name.

static _: () = (); //~ ERROR expected identifier, found reserved identifier `_`
struct _(); //~ ERROR expected identifier, found reserved identifier `_`
enum _ {} //~ ERROR expected identifier, found reserved identifier `_`
fn _() {} //~ ERROR expected identifier, found reserved identifier `_`
mod _ {} //~ ERROR expected identifier, found reserved identifier `_`
type _ = (); //~ ERROR expected identifier, found reserved identifier `_`
use _; //~ ERROR expected identifier, found reserved identifier `_`
use _ as g; //~ ERROR expected identifier, found reserved identifier `_`
trait _ {} //~ ERROR expected identifier, found reserved identifier `_`
trait _ = Copy; //~ ERROR expected identifier, found reserved identifier `_`
macro_rules! _ { () => {} } //~ ERROR expected identifier, found reserved identifier `_`
union _ { f: u8 } //~ ERROR expected one of `!` or `::`, found reserved identifier `_`

fn main() {}
