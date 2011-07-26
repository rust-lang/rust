


// -*- rust -*-
type clam = rec(@int x, @int y);

type fish = rec(@int a);

fn main() {
    let clam a = rec(x=@1, y=@2);
    let clam b = rec(x=@10, y=@20);
    let int z = a.x + b.y;
    log z;
    assert (z == 21);
    let fish forty = rec(a=@40);
    let fish two = rec(a=@2);
    let int answer = forty.a + two.a;
    log answer;
    assert (answer == 42);
}