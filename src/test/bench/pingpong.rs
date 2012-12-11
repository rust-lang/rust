// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Compare bounded and unbounded protocol performance.

extern mod std;

use pipes::{spawn_service, recv};
use std::time::precise_time_s;

proto! pingpong (
    ping: send {
        ping -> pong
    }

    pong: recv {
        pong -> ping
    }
)

proto! pingpong_unbounded (
    ping: send {
        ping -> pong
    }

    pong: recv {
        pong -> ping
    }

    you_will_never_catch_me: send {
        never_ever_ever -> you_will_never_catch_me
    }
)

// This stuff should go in libcore::pipes
macro_rules! move_it (
    { $x:expr } => { let t = move *ptr::addr_of(&($x)); move t }
)

macro_rules! follow (
    {
        $($message:path($($x: ident),+) -> $next:ident $e:expr)+
    } => (
        |m| match move m {
            $(Some($message($($x,)* move next)) => {
                let $next = move next;
                move $e })+
                _ => { fail }
        }
    );

    {
        $($message:path -> $next:ident $e:expr)+
    } => (
        |m| match move m {
            $(Some($message(move next)) => {
                let $next = move next;
                move $e })+
                _ => { fail }
        }
    )
)

fn switch<T: Send, Tb: Send, U>(+endp: pipes::RecvPacketBuffered<T, Tb>,
                      f: fn(+v: Option<T>) -> U) -> U {
    f(pipes::try_recv(move endp))
}

// Here's the benchmark

fn bounded(count: uint) {
    use pingpong::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = move ch;
        while count > 0 {
            ch = switch(move ch, follow! (
                ping -> next { server::pong(move next) }
            ));

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(move ch);

        ch = switch(move ch_, follow! (
            pong -> next { move next }
        ));

        count -= 1;
    }
}

fn unbounded(count: uint) {
    use pingpong_unbounded::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = move ch;
        while count > 0 {
            ch = switch(move ch, follow! (
                ping -> next { server::pong(move next) }
            ));

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(move ch);

        ch = switch(move ch_, follow! (
            pong -> next { move next }
        ));

        count -= 1;
    }
}

fn timeit(f: fn()) -> float {
    let start = precise_time_s();
    f();
    let stop = precise_time_s();
    stop - start
}

fn main() {
    let count = if os::getenv(~"RUST_BENCH").is_some() {
        250000
    } else {
        100
    };
    let bounded = do timeit { bounded(count) };
    let unbounded = do timeit { unbounded(count) };

    io::println(fmt!("count: %?\n", count));
    io::println(fmt!("bounded:   %? s\t(%? μs/message)",
                     bounded, bounded * 1000000. / (count as float)));
    io::println(fmt!("unbounded: %? s\t(%? μs/message)",
                     unbounded, unbounded * 1000000. / (count as float)));

    io::println(fmt!("\n\
                      bounded is %?%% faster",
                     (unbounded - bounded) / bounded * 100.));
}
