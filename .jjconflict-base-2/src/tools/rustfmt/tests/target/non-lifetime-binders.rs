fn main()
where
    for<'a, T: Sized + 'a, const C: usize> [&'a T; C]: Sized,
{
    let x = for<T> || {};

    let y: dyn for<T> Into<T>;

    let z: for<T> fn(T);
}
