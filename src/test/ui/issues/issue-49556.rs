// compile-pass
fn iter<'a>(data: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
    data.iter()
        .map(
            |x| x // fn(&'a usize) -> &'(ReScope) usize
        )
        .map(
            |x| *x // fn(&'(ReScope) usize) -> usize
        )
}

fn main() {
}
