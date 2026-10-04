//@ check-pass
#[allow(unused)]

fn test<T, F, U>(f: F) -> Box<dyn Fn(T) -> U + 'static>
where
    F: 'static + Fn(T) -> U,
    for<'a> U: 'a, // < This is the problematic line, see #97607
{
    Box::new(move |t| f(t))
}

fn main() {}
