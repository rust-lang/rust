// error-pattern: import

use m::unexported;

mod m {
    export exported;

    fn exported() { }

    fn unexported() { }
}


fn main() { unexported(); }
