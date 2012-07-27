// Compare bounded and unbounded protocol performance.

// xfail-pretty

use std;

import pipes::{spawn_service, recv};
import std::time::precise_time_s;

proto! pingpong {
    ping: send {
        ping -> pong
    }
    
    pong: recv {
        pong -> ping
    }
}

proto! pingpong_unbounded {
    ping: send {
        ping -> pong
    }
    
    pong: recv {
        pong -> ping
    }

    you_will_never_catch_me: send {
        never_ever_ever -> you_will_never_catch_me
    }
}

// This stuff should go in libcore::pipes
macro_rules! move {
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
}

macro_rules! follow {
    { 
        $($message:path($($x: ident),+) -> $next:ident $e:expr)+
    } => (
        |m| alt move(m) {
          $(some($message($($x,)* next)) {
            let $next = move!{next};
            $e })+
          _ { fail }
        }
    );

    { 
        $($message:path -> $next:ident $e:expr)+
    } => (
        |m| alt move(m) {
            $(some($message(next)) {
                let $next = move!{next};
                $e })+
                _ { fail }
        } 
    )
}

fn switch<T: send, Tb: send, U>(+endp: pipes::recv_packet_buffered<T, Tb>,
                      f: fn(+option<T>) -> U) -> U {
    f(pipes::try_recv(endp))
}

fn move<T>(-x: T) -> T { x }

// Here's the benchmark

fn bounded(count: uint) {
    import pingpong::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = ch;
        while count > 0 {
            ch = switch(ch, follow! {
                ping -> next { server::pong(next) }
            });

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(ch);

        ch = switch(ch_, follow! {
            pong -> next { next }
        });

        count -= 1;
    }
}

fn unbounded(count: uint) {
    import pingpong_unbounded::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = ch;
        while count > 0 {
            ch = switch(ch, follow! {
                ping -> next { server::pong(next) }
            });

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(ch);

        ch = switch(ch_, follow! {
            pong -> next { next }
        });

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
    let count = 1000000;
    let bounded = do timeit { bounded(count) };
    let unbounded = do timeit { unbounded(count) };

    io::println(#fmt("count: %?\n", count));
    io::println(#fmt("bounded:   %? s\t(%? μs/message)",
                     bounded, bounded * 1000000. / (count as float)));
    io::println(#fmt("unbounded: %? s\t(%? μs/message)",
                     unbounded, unbounded * 1000000. / (count as float)));

    io::println(#fmt("\n\
                      bounded is %?%% faster",
                     (unbounded - bounded) / bounded * 100.));
}
