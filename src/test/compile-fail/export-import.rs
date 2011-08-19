// error-pattern: unresolved import

import m::unexported;

mod m {
    export exported;

    fn exported() { }

    fn unexported() { }
}


fn main() { unexported(); }
