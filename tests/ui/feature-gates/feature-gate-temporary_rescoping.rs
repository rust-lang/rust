use std::temporary_rescoping;
//~^ ERROR: use of unstable library feature `temporary_rescoping`
use temporary_rescoping::{scope, extend};
//~^ ERROR: use of unstable library feature `temporary_rescoping`
//~| ERROR: use of unstable library feature `temporary_rescoping`

fn main() {
    'a: {
        scope!('a => 0);
        //~^ ERROR: use of unstable library feature `temporary_rescoping`
        //~| ERROR: FIXME: rescoping operators are not implemented
        extend!('a => 0);
        //~^ ERROR: use of unstable library feature `temporary_rescoping`
        //~| ERROR: FIXME: rescoping operators are not implemented
    }
}
