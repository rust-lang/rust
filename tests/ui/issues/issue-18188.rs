//@ check-pass

pub trait Promisable: Send + Sync {}
impl<T: Send + Sync> Promisable for T {}

pub fn propagate<'a, T, E, F, G>(mut action: F)
    -> Box<dyn FnMut(Result<T, E>) -> Result<T, E> + 'a>
    where
        T: Promisable + Clone + 'a,
        E: Promisable + Clone + 'a,
        F: FnMut(&T) -> Result<T, E> + Send + 'a,
        G: FnMut(Result<T, E>) -> Result<T, E> + 'a {
    Box::new(move |result: Result<T, E>| {
        match result {
            Ok(ref t) => action(t),
            Err(ref e) => Err(e.clone()),
        }
    })
}

fn main() {}
