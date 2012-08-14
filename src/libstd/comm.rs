/*!

Higher level communication abstractions.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

// Make sure we follow the new conventions
#[forbid(non_camel_case_types)];

import pipes::{channel, recv, chan, port, selectable};

export DuplexStream;

/// An extension of `pipes::stream` that allows both sending and receiving.
struct DuplexStream<T: send, U: send> : channel<T>, recv<U>, selectable {
    priv chan: chan<T>;
    priv port: port<U>;

    fn send(+x: T) {
        self.chan.send(x)
    }

    fn try_send(+x: T) -> bool {
        self.chan.try_send(x)
    }

    fn recv() -> U {
        self.port.recv()
    }

    fn try_recv() -> option<U> {
        self.port.try_recv()
    }

    pure fn peek() -> bool {
        self.port.peek()
    }

    pure fn header() -> *pipes::packet_header {
        self.port.header()
    }
}

/// Creates a bidirectional stream.
fn DuplexStream<T: send, U: send>()
    -> (DuplexStream<T, U>, DuplexStream<U, T>)
{
    let (c2, p1) = pipes::stream();
    let (c1, p2) = pipes::stream();
    (DuplexStream {
        chan: c1,
        port: p1
    },
     DuplexStream {
         chan: c2,
         port: p2
     })
}

#[cfg(test)]
mod test {
    #[test]
    fn DuplexStream1() {
        let (left, right) = DuplexStream();

        left.send(~"abc");
        right.send(123);

        assert left.recv() == 123;
        assert right.recv() == ~"abc";
    }
}
