use std;
use std::arc;
use comm::*;

fn main() {
    let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = arc::ARC(v);

    do task::spawn() |move arc_v| { //~ NOTE move of variable occurred here
        let v = *arc::get(&arc_v);
        assert v[3] == 4;
    };

    assert (*arc::get(&arc_v))[2] == 3; //~ ERROR use of moved variable: `arc_v`

    log(info, arc_v);
}
