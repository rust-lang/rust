extern mod std;
use comm::Chan;
use comm::Port;
use comm::recv;
use comm::send;

enum msg { closed, received(~[u8]), }

fn producer(c: Chan<~[u8]>) {
    send(c, ~[1u8, 2u8, 3u8, 4u8]);
    let empty: ~[u8] = ~[];
    send(c, empty);
}

fn packager(cb: Chan<Chan<~[u8]>>, msg: Chan<msg>) {
    let p: Port<~[u8]> = Port();
    send(cb, Chan(p));
    loop {
        debug!("waiting for bytes");
        let data = recv(p);
        debug!("got bytes");
        if vec::len(data) == 0u {
            debug!("got empty bytes, quitting");
            break;
        }
        debug!("sending non-empty buffer of length");
        log(debug, vec::len(data));
        send(msg, received(data));
        debug!("sent non-empty buffer");
    }
    debug!("sending closed message");
    send(msg, closed);
    debug!("sent closed message");
}

fn main() {
    let p: Port<msg> = Port();
    let ch = Chan(p);
    let recv_reader: Port<Chan<~[u8]>> = Port();
    let recv_reader_chan = Chan(recv_reader);
    let pack = task::spawn(|| packager(recv_reader_chan, ch) );

    let source_chan: Chan<~[u8]> = recv(recv_reader);
    let prod = task::spawn(|| producer(source_chan) );

    loop {
        let msg = recv(p);
        match msg {
          closed => { debug!("Got close message"); break; }
          received(data) => {
            debug!("Got data. Length is:");
            log(debug, vec::len::<u8>(data));
          }
        }
    }
}
