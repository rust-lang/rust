fn main() {
    let p = comm::port::<uint>();
    let ch = comm::chan(p);

    let x = ~1;
    let x_in_parent = ptr::addr_of(*x) as uint;

    let y = ~2;
    let y_in_parent = ptr::addr_of(*y) as uint;

    task::spawn(fn~[copy ch, y; move x]() {
        let x_in_child = ptr::addr_of(*x) as uint;
        comm::send(ch, x_in_child);

        let y_in_child = ptr::addr_of(*y) as uint;
        comm::send(ch, y_in_child);
    });

    let x_in_child = comm::recv(p);
    assert x_in_parent == x_in_child;

    let y_in_child = comm::recv(p);
    assert y_in_parent != y_in_child;
}
