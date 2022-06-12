// check-pass

fn test<T, F, U>(f: F) -> Box<dyn Fn(T) -> U + 'static>
where
    F: 'static + Fn(T) -> U,
    for<'a> U: 'a, // < This is the problematic line -- remove it, and it passes.
{
    Box::new(move |t| f(t))
}

fn main() {}
