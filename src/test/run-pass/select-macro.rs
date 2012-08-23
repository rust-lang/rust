// xfail-test - this isn't really a test.

 {

// select!
macro_rules! select_if {

    {
        $index:expr,
        $count:expr
    } => {
        fail
    };

    {
        $index:expr,
        $count:expr,
        $port:path => [
            $(type_this $message:path$(($(x $x: ident),+))dont_type_this*
              -> $next:ident => { $e:expr }),+
        ]
        $(, $ports:path => [
            $(type_this $messages:path$(($(x $xs: ident),+))dont_type_this*
              -> $nexts:ident => { $es:expr }),+
        ] )*
    } => {
        if $index == $count {
            match move pipes::try_recv($port) {
              $(some($message($($(ref $x,)+)* ref next)) => {
                // FIXME (#2329) we really want move out of enum here.
                let $next = unsafe { let x <- *ptr::addr_of(*next); x };
                $e
              })+
              _ => fail
            }
        } else {
            select_if!(
                $index,
                $count + 1
                $(, $ports => [
                    $(type_this $messages$(($(x $xs),+))dont_type_this*
                      -> $nexts => { $es }),+
                ])*
            )
        }
    };
}

macro_rules! select {
    {
        $( $port:path => {
            $($message:path$(($($x: ident),+))dont_type_this*
              -> $next:ident $e:expr),+
        } )+
    } => {
        let index = pipes::selecti([$(($port).header()),+]/_);
        select_if!(index, 0 $(, $port => [
            $(type_this $message$(($(x $x),+))dont_type_this* -> $next => { $e }),+
        ])+)
    }
}

}
