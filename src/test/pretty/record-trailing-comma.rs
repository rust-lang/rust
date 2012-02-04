// pp-exact
type thing = {x: int, y: int,};

fn main() {
    let sth = {x: 0, y: 1,};
    let sth2 = {y: 9 with sth};
    assert sth.x + sth2.y == 9;
}
