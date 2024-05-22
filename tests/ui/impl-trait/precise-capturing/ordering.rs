#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn lt<'a>() -> impl use<'a, 'a> Sized {}
//~^ ERROR cannot capture parameter `'a` twice

fn ty<T>() -> impl use<T, T> Sized {}
//~^ ERROR cannot capture parameter `T` twice

fn ct<const N: usize>() -> impl use<N, N> Sized {}
//~^ ERROR cannot capture parameter `N` twice

fn ordering<'a, T>() -> impl use<T, 'a> Sized {}
//~^ ERROR lifetime parameter `'a` must be listed before non-lifetime parameters

fn main() {}
