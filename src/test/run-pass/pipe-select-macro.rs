// xfail-pretty

// Protocols
proto! foo {
    foo:recv {
        do_foo -> foo
    }
}

proto! bar {
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
}

fn macros() {
    include!("select-macro.rs");
}

// Code
fn test(+foo: foo::client::foo, +bar: bar::client::bar) {
    import bar::do_baz;

    select! {
        foo => {
            foo::do_foo -> _next {
            }
        }

        bar => {
            bar::do_bar(x) -> _next {
                debug!("%?", x)
            },

            do_baz(b) -> _next {
                if *b { debug!("true") } else { debug!("false") }
            }
        }
    }
}

fn main() {
}
