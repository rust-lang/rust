trait Tailed<'a>: 'a {
    type Tail: Tailed<'a>;
}

struct List<'a, T: Tailed<'a>> {
    //~^ ERROR overflow computing implied lifetime bounds for `List`
    next: Box<List<'a, T::Tail>>,
    node: &'a T,
}

fn main() {}
