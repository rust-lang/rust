// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME #7303: xfail-test

// Protocols
proto! foo (
    foo:recv {
        do_foo -> foo
    }
)

proto! bar (
    bar:recv {
        do_bar(int) -> barbar,
        do_baz(bool) -> bazbar,
    }

    barbar:send {
        rebarbar -> bar,
    }

    bazbar:send {
        rebazbar -> bar
    }
)

fn macros() {
    include!("select-macro.rs");
}

// Code
fn test(+foo: foo::client::foo, +bar: bar::client::bar) {
    use bar::do_baz;

    select! (
        foo => {
            foo::do_foo -> _next {
            }
        }

        bar => {
            bar::do_bar(x) -> _next {
                debug!("%?", x)
            },

            do_baz(b) -> _next {
                if b { debug!("true") } else { debug!("false") }
            }
        }
    )
}

pub fn main() {
}
