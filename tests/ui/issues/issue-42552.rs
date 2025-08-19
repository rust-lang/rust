//@ run-pass
// Regression test for an obscure issue with the projection cache.

fn into_iter<I: Iterator>(a: &I) -> Groups<'_, I> {
    Groups { _a: a }
}

pub struct Groups<'a, I: 'a> {
    _a: &'a I,
}

impl<'a, I: Iterator> Iterator for Groups<'a, I> {
    type Item = Group<'a, I>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

pub struct Group<'a, I: Iterator + 'a>
    where I::Item: 'a       // <-- needed to trigger ICE!
{
    _phantom: &'a (),
    _ice_trigger: I::Item,  // <-- needed to trigger ICE!
}


fn main() {
    let _ = into_iter(&[0].iter().map(|_| 0)).map(|grp| {
        let _g = grp;
    });
}
