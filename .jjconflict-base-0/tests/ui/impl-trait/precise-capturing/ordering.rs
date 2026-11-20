fn lt<'a>() -> impl Sized + use<'a, 'a> {}
//~^ ERROR cannot capture parameter `'a` twice

fn ty<T>() -> impl Sized + use<T, T> {}
//~^ ERROR cannot capture parameter `T` twice

fn ct<const N: usize>() -> impl Sized + use<N, N> {}
//~^ ERROR cannot capture parameter `N` twice

fn ordering<'a, T>() -> impl Sized + use<T, 'a> {}
//~^ ERROR lifetime parameter `'a` must be listed before non-lifetime parameters

fn main() {}
