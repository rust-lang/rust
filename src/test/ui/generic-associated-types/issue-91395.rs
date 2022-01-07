// check-fail
// FIXME(generic_associated_types): This *should* pass, but fails
// because we pick some region `'_#1r` instead of `'a`

#![feature(generic_associated_types)]

trait Foo {
    type Item;
    fn item(&self) -> Self::Item;
}

trait Table {
    type Blocks<'a>: Foo
    where
        Self: 'a;

    fn blocks(&self) -> Self::Blocks<'_>;
}

fn box_static_block<U: 'static>(_block: U) {}

fn clone_static_table<'a, T>(table: &'a T)
where
    T: Table,
    <<T as Table>::Blocks<'a> as Foo>::Item: 'static,
    //for<'x> <<T as Table>::Blocks<'x> as Foo>::Item: 'static, // This also doesn't work
{
    let block = table.blocks().item();
    box_static_block(block);
    //~^ the associated type
}

fn main() {}
