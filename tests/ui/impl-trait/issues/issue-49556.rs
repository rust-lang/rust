//@ check-pass
fn iter<'a>(data: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
    data.iter()
        .map(
            |x| x // fn(&'a usize) -> &'a usize
        )
        .map(
            |x| *x // fn(&'a usize) -> usize
        )
}

fn main() {
}
