use std;
import std::arc;
import comm::*;

fn main() {
    let v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = arc::arc(v);

    let p = port();
    let c = chan(p);
    
    task::spawn() {||
        let p = port();
        c.send(chan(p));

        let arc_v = p.recv();

        let v = *arc::get::<[int]>(&arc_v);
        assert v[3] == 4;
    };

    let c = p.recv();
    c.send(arc::clone(&arc_v));

    assert (*arc::get(&arc_v))[2] == 3;

    log(info, arc_v);
}
