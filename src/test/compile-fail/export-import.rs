// error-pattern: import

use m::unexported;

mod m {
    #[legacy_exports];
    export exported;

    fn exported() { }

    fn unexported() { }
}


fn main() { unexported(); }
