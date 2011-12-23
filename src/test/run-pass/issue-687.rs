use std;
import vec;
import task;
import comm;
import comm::chan;
import comm::port;
import comm::recv;
import comm::send;

tag msg { closed; received([u8]); }

fn producer(c: chan<[u8]>) {
    send(c, [1u8, 2u8, 3u8, 4u8]);
    let empty: [u8] = [];
    send(c, empty);
}

fn packager(&&args: (chan<chan<[u8]>>, chan<msg>)) {
    let (cb, msg) = args;
    let p: port<[u8]> = port();
    send(cb, chan(p));
    while true {
        #debug("waiting for bytes");
        let data = recv(p);
        #debug("got bytes");
        if vec::len(data) == 0u {
            #debug("got empty bytes, quitting");
            break;
        }
        #debug("sending non-empty buffer of length");
        log(debug, vec::len(data));
        send(msg, received(data));
        #debug("sent non-empty buffer");
    }
    #debug("sending closed message");
    send(msg, closed);
    #debug("sent closed message");
}

fn main() {
    let p: port<msg> = port();
    let recv_reader: port<chan<[u8]>> = port();
    let pack = task::spawn((chan(recv_reader), chan(p)), packager);

    let source_chan: chan<[u8]> = recv(recv_reader);
    let prod = task::spawn(source_chan, producer);

    while true {
        let msg = recv(p);
        alt msg {
          closed. { #debug("Got close message"); break; }
          received(data) {
            #debug("Got data. Length is:");
            log(debug, vec::len::<u8>(data));
          }
        }
    }
}
