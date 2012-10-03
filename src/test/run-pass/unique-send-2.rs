extern mod std;

fn child(c: comm::Chan<~uint>, i: uint) {
    comm::send(c, ~i);
}

fn main() {
    let p = comm::Port();
    let ch = comm::Chan(&p);
    let n = 100u;
    let mut expected = 0u;
    for uint::range(0u, n) |i| {
        task::spawn(|| child(ch, i) );
        expected += i;
    }

    let mut actual = 0u;
    for uint::range(0u, n) |_i| {
        let j = comm::recv(p);
        actual += *j;
    }

    assert expected == actual;
}