// xfail-test
// error-pattern: instantiating a type parameter with an incompatible type
extern mod std;
use std::arc::rw_arc;

fn main() {
    let arc1  = ~rw_arc(true);
    let _arc2 = ~rw_arc(arc1);
}
