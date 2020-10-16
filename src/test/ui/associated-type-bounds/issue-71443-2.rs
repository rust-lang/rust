// check-pass

#![feature(associated_type_bounds)]

fn hello<'b, F>()
where
    for<'a> F: Iterator<Item: 'a> + 'b,
{
}

fn main() {}
