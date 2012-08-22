fn main() {
    let (c,p) = pipes::stream();
    let x = some(p);
    c.send(false);
    match move x {
        some(move z) if z.recv() => { fail }, //~ ERROR cannot bind by-move into a pattern guard
        some(move z) => { assert !z.recv(); },
        none => fail
    }
}
