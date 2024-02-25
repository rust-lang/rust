//@ revisions: cfail
#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused_braces, unused_variables)]

trait Delegates<T> {}

struct FileCap<const Op: bool> {}

fn writes_to_path<C>(cap: &C)
where
    C: Delegates<FileCap<{ false }>>,
{
    writes_to_specific_path(&cap);
    //~^ error: the trait bound
}

fn writes_to_specific_path<C>(cap: &C)
where
    C: Delegates<FileCap<{ false }>>,
{
}

fn main() {}
