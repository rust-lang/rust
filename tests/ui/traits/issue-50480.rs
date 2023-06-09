#[derive(Clone, Copy)]
//~^ ERROR the trait `Copy` cannot be implemented for this type
struct Foo(N, NotDefined, <i32 as Iterator>::Item, Vec<i32>, String);
//~^ ERROR cannot find type `NotDefined` in this scope
//~| ERROR cannot find type `NotDefined` in this scope
//~| ERROR cannot find type `N` in this scope
//~| ERROR cannot find type `N` in this scope

#[derive(Clone, Copy)]
//~^ ERROR the trait `Copy` cannot be implemented for this type
struct Bar<T>(T, N, NotDefined, <i32 as Iterator>::Item, Vec<i32>, String);
//~^ ERROR cannot find type `NotDefined` in this scope
//~| ERROR cannot find type `N` in this scope

fn main() {}
