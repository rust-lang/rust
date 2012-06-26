#[doc="A new implementation of communication.

This should be implementing almost entirely in Rust, and hopefully
avoid needing a single global lock."]

import arc::methods;
import dvec::dvec;
import dvec::{extensions};

export port;
export chan;
export send, recv;
export methods;

type raw_port<T: send> = arc::exclusive<dvec<T>>;

enum port<T: send> {
    port_(raw_port<T>)
}
enum chan<T: send> {
    chan_(raw_port<T>)
}

fn port<T: send>() -> port<T> {
    port_(arc::exclusive(dvec()))
}

fn chan<T: send>(p: port<T>) -> chan<T> {
    chan_((*p).clone())
}

fn send<T: send>(c: chan<T>, -x: T) {
    let mut x <- some(x);
    do (*c).with {|cond, data|
        let mut xx = none;
        xx <-> x;
        (*data).push(option::unwrap(xx));
        cond.signal();
    }
}

fn recv<T: send>(p: port<T>) -> T {
    do (*p).with {|cond, data|
        if (*data).len() == 0u {
            cond.wait();
        }
        assert (*data).len() > 0u;
        (*data).shift()
    }
}

impl methods<T: send> for chan<T> {
    fn send(-x: T) {
        send(self, x)
    }

    fn clone() -> chan<T> {
        chan_((*self).clone())
    }
}

impl methods<T: send> for port<T> {
    fn recv() -> T {
        recv(self)
    }

    fn chan() -> chan<T> {
        chan(self)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn newport_simple() {
        let p = port();
        let c = chan(p);

        c.send(42);
        assert p.recv() == 42;
    }
}
