


// -*- rust -*-
type clam = rec(@int x, @int y);

type fish = tup(@int);

fn main() {
    let clam a = rec(x=@1, y=@2);
    let clam b = rec(x=@10, y=@20);
    let int z = a.x + b.y;
    log z;
    assert (z == 21);
    let fish forty = tup(@40);
    let fish two = tup(@2);
    let int answer = forty._0 + two._0;
    log answer;
    assert (answer == 42);
}