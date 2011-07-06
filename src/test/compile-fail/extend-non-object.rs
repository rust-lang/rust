//error-pattern:x does not have object type
use std;

fn main() {
    auto x = 3;

    auto anon_obj = obj {
        fn foo() -> int {
            ret 3;
        }
        with x
    };
}
