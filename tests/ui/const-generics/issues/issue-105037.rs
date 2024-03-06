//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(dead_code)]

trait Table<const D: usize>: Sync {
    const COLUMNS: usize;
}

struct Table1<const D: usize>;
impl<const D: usize> Table<D> for Table1<D> {
    const COLUMNS: usize = 123;
}

struct Table2<const D: usize>;
impl<const D: usize> Table<D> for Table2<D> {
    const COLUMNS: usize = 456;
}

fn process_table<T: Table<D>, const D: usize>(_table: T)
where
    [(); T::COLUMNS]:,
{
}

fn process_all_tables<const D: usize>()
where
    [(); Table2::<D>::COLUMNS]:,
    [(); Table1::<D>::COLUMNS]:,
{
    process_table(Table1::<D>);
    process_table(Table2::<D>);
}

fn main() {}
