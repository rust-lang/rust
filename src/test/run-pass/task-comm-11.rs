// xfail-fast
#[legacy_modes];

extern mod std;

fn start(c: pipes::Chan<pipes::Chan<int>>) {
    let (ch, p) = pipes::stream();
    c.send(move ch);
}

fn main() {
    let (ch, p) = pipes::stream();
    let child = task::spawn(|move ch| start(ch) );
    let c = p.recv();
}
