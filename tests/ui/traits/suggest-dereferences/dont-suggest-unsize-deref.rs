fn use_iterator<I>(itr: I)
where
    I: IntoIterator<Item = i32>,
{
}

fn pass_iterator<I>(i: &dyn IntoIterator<Item = i32, IntoIter = I>)
where
    I: Iterator<Item = i32>,
{
    use_iterator(i);
    //~^ ERROR `&dyn IntoIterator<IntoIter = I, Item = i32>` is not an iterator
}

fn main() {}
