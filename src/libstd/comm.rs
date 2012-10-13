/*!

Higher level communication abstractions.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];

use pipes::{Channel, Recv, Chan, Port, Selectable};

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<T: Send, U: Send> {
    priv chan: Chan<T>,
    priv port: Port <U>,
}

impl<T: Send, U: Send> DuplexStream<T, U> : Channel<T> {
    fn send(x: T) {
        self.chan.send(move x)
    }

    fn try_send(x: T) -> bool {
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
pub fn DuplexStream<T: Send, U: Send>()
    -> (DuplexStream<T, U>, DuplexStream<U, T>)
{
    let (c2, p1) = pipes::stream();
    let (c1, p2) = pipes::stream();
    (DuplexStream {
        chan: move c1,
        port: move p1
    },
     DuplexStream {
         chan: move c2,
         port: move p2
     })
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    #[test]
    fn DuplexStream1() {
        let (left, right) = DuplexStream();

        left.send(~"abc");
        right.send(123);

        assert left.recv() == 123;
        assert right.recv() == ~"abc";
    }
}
