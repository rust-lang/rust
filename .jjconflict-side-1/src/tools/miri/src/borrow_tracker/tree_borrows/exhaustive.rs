//! Exhaustive testing utilities.
//! (These are used in Tree Borrows `#[test]`s for thorough verification
//! of the behavior of the state machine of permissions,
//! but the contents of this file are extremely generic)

pub trait Exhaustive: Sized {
    fn exhaustive() -> Box<dyn Iterator<Item = Self>>;
}

macro_rules! precondition {
    ($cond:expr) => {
        if !$cond {
            continue;
        }
    };
}
pub(crate) use precondition;

// Trivial impls of `Exhaustive` for the standard types with 0, 1 and 2 elements respectively.

impl Exhaustive for ! {
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::empty())
    }
}

impl Exhaustive for () {
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::once(()))
    }
}

impl Exhaustive for bool {
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(vec![true, false].into_iter())
    }
}

// Some container impls for `Exhaustive`.

impl<T> Exhaustive for Option<T>
where
    T: Exhaustive + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::once(None).chain(T::exhaustive().map(Some)))
    }
}

impl<T1, T2> Exhaustive for (T1, T2)
where
    T1: Exhaustive + Clone + 'static,
    T2: Exhaustive + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(T1::exhaustive().flat_map(|t1| T2::exhaustive().map(move |t2| (t1.clone(), t2))))
    }
}

impl<T> Exhaustive for [T; 1]
where
    T: Exhaustive + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(T::exhaustive().map(|t| [t]))
    }
}

impl<T> Exhaustive for [T; 2]
where
    T: Exhaustive + Clone + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(T::exhaustive().flat_map(|t1| T::exhaustive().map(move |t2| [t1.clone(), t2])))
    }
}

impl<T> Exhaustive for [T; 3]
where
    T: Exhaustive + Clone + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(
            <[T; 2]>::exhaustive()
                .flat_map(|[t1, t2]| T::exhaustive().map(move |t3| [t1.clone(), t2.clone(), t3])),
        )
    }
}

impl<T> Exhaustive for [T; 4]
where
    T: Exhaustive + Clone + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(<[T; 2]>::exhaustive().flat_map(|[t1, t2]| {
            <[T; 2]>::exhaustive().map(move |[t3, t4]| [t1.clone(), t2.clone(), t3, t4])
        }))
    }
}

impl<T> Exhaustive for [T; 5]
where
    T: Exhaustive + Clone + 'static,
{
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        Box::new(<[T; 2]>::exhaustive().flat_map(|[t1, t2]| {
            <[T; 3]>::exhaustive().map(move |[t3, t4, t5]| [t1.clone(), t2.clone(), t3, t4, t5])
        }))
    }
}
