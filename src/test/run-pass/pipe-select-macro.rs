// xfail-test

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


// select!
macro_rules! select_if {
    {
        $index:expr,
        $count:expr,
        $port:path => [
            $($message:path$(($($x: ident),+))dont_type_this*
              -> $next:ident $e:expr),+
        ],
        $( $ports:path => [
            $($messages:path$(($($xs: ident),+))dont_type_this*
              -> $nexts:ident $es:expr),+
        ], )*
    } => {
        log_syntax!{select_if1};
        if $index == $count {
            alt move pipes::try_recv($port) {
              $(some($message($($($x,)+)* next)) => {
                // FIXME (#2329) we really want move out of enum here.
                let $next = unsafe { let x <- *ptr::addr_of(next); x };
                $e
              })+
              _ => fail
            }
        } else {
            select_if!{
                $index,
                $count + 1,
                $( $ports => [
                    $($messages$(($($xs),+))dont_type_this*
                      -> $nexts $es),+
                ], )*
            }
        }
    };

    {
        $index:expr,
        $count:expr,
    } => {
        log_syntax!{select_if2};
        fail
    }
}

macro_rules! select {
    {
        $( $port:path => {
            $($message:path$(($($x: ident),+))dont_type_this*
              -> $next:ident $e:expr),+
        } )+
    } => {
        let index = pipes::selecti([$(($port).header()),+]/_);
        log_syntax!{select};
        log_syntax!{
        select_if!{index, 0, $( $port => [
            $($message$(($($x),+))dont_type_this* -> $next $e),+
        ], )+}
        };
        select_if!{index, 0, $( $port => [
            $($message$(($($x),+))dont_type_this* -> $next $e),+
        ], )+}
    }
}

// Code
fn test(+foo: foo::client::foo, +bar: bar::client::bar) {
    select! {
        foo => {
            foo::do_foo -> _next {
            }
        }

        bar => {
            bar::do_bar(x) -> _next {
                //debug!("%?", x)
            },

            do_baz(b) -> _next {
                //if b { debug!("true") } else { debug!("false") }
            }
        }
    }
}

fn main() {
}
