/*!

Higher level communication abstractions.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use pipes::{Channel, Recv, Chan, Port, Selectable};

export DuplexStream;

/// An extension of `pipes::stream` that allows both sending and receiving.
struct DuplexStream<T: Send, U: Send> {
    priv chan: Chan<T>,
    priv port: Port <U>,
}

impl<T: Send, U: Send> DuplexStream<T, U> : Channel<T> {
    fn send(+x: T) {
        self.chan.send(move x)
    }

    fn try_send(+x: T) -> bool {
        self.chan.try_send(move x)
    }
}

impl<T: Send, U: Send> DuplexStream<T, U> : Recv<U> {
    fn recv() -> U {
        self.port.recv()
    }

    fn try_recv() -> Option<U> {
        self.port.try_recv()
    }

    pure fn peek() -> bool {
        self.port.peek()
    }
}

impl<T: Send, U: Send> DuplexStream<T, U> : Selectable {
    pure fn header() -> *pipes::PacketHeader {
        self.port.header()
    }
}

/// Creates a bidirectional stream.
fn DuplexStream<T: Send, U: Send>()
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
