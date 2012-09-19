// xfail-fast
#[legacy_modes];

fn main() {
    let po = pipes::PortSet();

    // Spawn 10 tasks each sending us back one int.
    let mut i = 10;
    while (i > 0) {
        log(debug, i);
        let (ch, p) = pipes::stream();
        po.add(p);
        task::spawn(|copy i| child(i, ch) );
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    while (i > 0) {
        log(debug, i);
        po.recv();
        i = i - 1;
    }

    debug!("main thread exiting");
}

fn child(x: int, ch: pipes::Chan<int>) {
    log(debug, x);
    ch.send(x);
}
