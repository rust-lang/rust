// error-pattern: unsatisfied precondition constraint
use std;
import std::arc;
import comm::*;

fn main() {
    let v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = arc::arc(v);
    
    task::spawn() {|move arc_v|
        let v = *arc::get(&arc_v);
        assert v[3] == 4;
    };

    assert (*arc::get(&arc_v))[2] == 3;

    log(info, arc_v);
}
