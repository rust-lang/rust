//@ check-pass

pub fn iter<'a>(v: Vec<(u32, &'a u32)>) -> impl DoubleEndedIterator<Item = (u32, &u32)> {
    //~^ WARNING lifetime flowing from input to output with different syntax
    v.into_iter()
}

fn main() {}
