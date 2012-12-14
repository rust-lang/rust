// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
/**
   A test case for issue #577, which also exposes #588
*/

extern mod std;

fn child() { }

struct notify {
    ch: oldcomm::Chan<bool>, v: @mut bool,
}

impl notify : Drop {
    fn finalize(&self) {
        error!("notify: task=%? v=%x unwinding=%b b=%b",
               task::get_task(),
               ptr::addr_of(&(*(self.v))) as uint,
               task::failing(),
               *(self.v));
        let b = *(self.v);
        oldcomm::send(self.ch, b);
    }
}

fn notify(ch: oldcomm::Chan<bool>, v: @mut bool) -> notify {
    notify {
        ch: ch,
        v: v
    }
}

fn joinable(+f: fn~()) -> oldcomm::Port<bool> {
    fn wrapper(+c: oldcomm::Chan<bool>, +f: fn()) {
        let b = @mut false;
        error!("wrapper: task=%? allocated v=%x",
               task::get_task(),
               ptr::addr_of(&(*b)) as uint);
        let _r = notify(c, b);
        f();
        *b = true;
    }
    let p = oldcomm::Port();
    let c = oldcomm::Chan(&p);
    do task::spawn_unlinked { wrapper(c, copy f) };
    p
}

fn join(port: oldcomm::Port<bool>) -> bool {
    oldcomm::recv(port)
}

fn main() {
    // tasks
    let t1;
    let t2;

    let c1 = child, c2 = child;
    t1 = joinable(c1);
    t2 = joinable(c2);

    assert (t1 == t1);
    assert (t1 != t2);

    // ports
    let p1;
    let p2;

    p1 = oldcomm::Port::<int>();
    p2 = oldcomm::Port::<int>();

    assert (p1 == p1);
    assert (p1 != p2);

    // channels
    let c1 = oldcomm::Chan(p1);
    let c2 = oldcomm::Chan(p2);

    assert (c1 == c1);
    assert (c1 != c2);

    join(t1);
    join(t2);
}
