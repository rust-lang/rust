// Compare bounded and unbounded protocol performance.

// xfail-pretty

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
    { $x:expr } => { let t <- *ptr::addr_of(&($x)); t }
)

macro_rules! follow (
    {
        $($message:path($($x: ident),+) -> $next:ident $e:expr)+
    } => (
        |m| match move m {
            $(Some($message($($x,)* move next)) => {
                let $next = next;
                $e })+
                _ => { fail }
        }
    );

    {
        $($message:path -> $next:ident $e:expr)+
    } => (
        |m| match move m {
            $(Some($message(move next)) => {
                let $next = next;
                $e })+
                _ => { fail }
        }
    )
)

fn switch<T: Send, Tb: Send, U>(+endp: pipes::RecvPacketBuffered<T, Tb>,
                      f: fn(+v: Option<T>) -> U) -> U {
    f(pipes::try_recv(endp))
}

// Here's the benchmark

fn bounded(count: uint) {
    use pingpong::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = ch;
        while count > 0 {
            ch = switch(ch, follow! (
                ping -> next { server::pong(next) }
            ));

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(ch);

        ch = switch(ch_, follow! (
            pong -> next { next }
        ));

        count -= 1;
    }
}

fn unbounded(count: uint) {
    use pingpong_unbounded::*;

    let mut ch = do spawn_service(init) |ch| {
        let mut count = count;
        let mut ch = ch;
        while count > 0 {
            ch = switch(ch, follow! (
                ping -> next { server::pong(next) }
            ));

            count -= 1;
        }
    };

    let mut count = count;
    while count > 0 {
        let ch_ = client::ping(ch);

        ch = switch(ch_, follow! (
            pong -> next { next }
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
