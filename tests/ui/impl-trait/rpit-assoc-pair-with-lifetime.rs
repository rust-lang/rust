//@ check-pass

pub fn iter<'a>(v: Vec<(u32, &'a u32)>) -> impl DoubleEndedIterator<Item = (u32, &u32)> {
    //~^ WARNING elided lifetime has a name
    v.into_iter()
}

fn main() {}
