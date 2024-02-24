// Regression test for #42602. It used to be that we had
// a dep-graph like
//
//     typeck(foo) -> FnOnce -> typeck(bar)
//
// This was fixed by improving the resolution of the `FnOnce` trait
// selection node.

//@ revisions:cfail1 cfail2 cfail3
//@ compile-flags:-Zquery-dep-graph
//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]

fn main() {
    a::foo();
    b::bar();
}

mod a {
    #[cfg(cfail1)]
    pub fn foo() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }

    #[cfg(not(cfail1))]
    pub fn foo() {
        let x = vec![1, 2, 3, 4];
        let v = || ::std::mem::drop(x);
        v();
    }
}

mod b {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    pub fn bar() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }
}
