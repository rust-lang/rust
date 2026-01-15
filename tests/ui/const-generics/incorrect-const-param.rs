// #84327

struct VecWrapper<T>(Vec<T>);

// Correct
impl<T, const N: usize> From<[T; N]> for VecWrapper<T>
where
    T: Clone,
{
    fn from(slice: [T; N]) -> Self {
        VecWrapper(slice.to_vec())
    }
}

// Forgot const
impl<T, N: usize> From<[T; N]> for VecWrapper<T> //~ ERROR expected value, found type parameter `N`
where //~^ ERROR expected trait, found builtin type `usize`
    T: Clone,
{
    fn from(slice: [T; N]) -> Self { //~ ERROR expected value, found type parameter `N`
        VecWrapper(slice.to_vec())
    }
}

// Forgot type
impl<T, const N> From<[T; N]> for VecWrapper<T> //~ ERROR expected `:`, found `>`
where
    T: Clone,
{
    fn from(slice: [T; N]) -> Self {
        VecWrapper(slice.to_vec())
    }
}

// Forgot const and type
impl<T, N> From<[T; N]> for VecWrapper<T> //~ ERROR expected value, found type parameter `N`
where
    T: Clone,
{
    fn from(slice: [T; N]) -> Self { //~ ERROR expected value, found type parameter `N`
        VecWrapper(slice.to_vec())
    }
}

fn main() {}
