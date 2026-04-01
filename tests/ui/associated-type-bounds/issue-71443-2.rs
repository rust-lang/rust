//@ check-pass

fn hello<'b, F>()
where
    for<'a> F: Iterator<Item: 'a> + 'b,
{
}

fn main() {}
