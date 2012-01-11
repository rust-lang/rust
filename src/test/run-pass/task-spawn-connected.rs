fn stringifier(from_par: comm::port<uint>,
               to_par: comm::chan<str>) {
    let value: uint;
    do {
        value = comm::recv(from_par);
        comm::send(to_par, uint::to_str(value, 10u));
    } while value != 0u;
}

fn main() {
    let t = task::spawn_connected(stringifier);
    comm::send(t.to_child, 22u);
    assert comm::recv(t.from_child) == "22";
    comm::send(t.to_child, 23u);
    assert comm::recv(t.from_child) == "23";
    comm::send(t.to_child, 0u);
    assert comm::recv(t.from_child) == "0";
}