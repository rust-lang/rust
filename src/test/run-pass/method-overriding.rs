// xfail-boot
// xfail-stage0
use std;
import std._vec.len;
fn main() {

    obj a() {
        fn foo() -> int {
            ret 2;
        }
        fn bar() -> int {
            ret self.foo();
        }
    }

    auto my_a = a();

    // Step 1 is to add support for this "with" syntax
    auto my_b = obj { 
        fn baz() -> int { 
            ret self.foo(); 
        } 
        with my_a 
    };
    
}
