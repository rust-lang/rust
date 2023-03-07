// run-pass

#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unknown_lints)]
// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]

struct A { a: isize, b: isize }
struct Abox { a: Box<isize>, b: Box<isize> }

fn ret_int_i() -> isize { 10 }

fn ret_ext_i() -> Box<isize> { Box::new(10) }

fn ret_int_rec() -> A { A {a: 10, b: 10} }

fn ret_ext_rec() -> Box<A> { Box::new(A {a: 10, b: 10}) }

fn ret_ext_mem() -> Abox { Abox {a: Box::new(10), b: Box::new(10) } }

fn ret_ext_ext_mem() -> Box<Abox> { Box::new(Abox{a: Box::new(10), b: Box::new(10) }) }

pub fn main() {
    let mut int_i: isize;
    let mut ext_i: Box<isize>;
    let mut int_rec: A;
    let mut ext_rec: Box<A>;
    let mut ext_mem: Abox;
    let mut ext_ext_mem: Box<Abox>;
    int_i = ret_int_i(); // initializing

    int_i = ret_int_i(); // non-initializing

    int_i = ret_int_i(); // non-initializing

    ext_i = ret_ext_i(); // initializing

    ext_i = ret_ext_i(); // non-initializing

    ext_i = ret_ext_i(); // non-initializing

    int_rec = ret_int_rec(); // initializing

    int_rec = ret_int_rec(); // non-initializing

    int_rec = ret_int_rec(); // non-initializing

    ext_rec = ret_ext_rec(); // initializing

    ext_rec = ret_ext_rec(); // non-initializing

    ext_rec = ret_ext_rec(); // non-initializing

    ext_mem = ret_ext_mem(); // initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

}
