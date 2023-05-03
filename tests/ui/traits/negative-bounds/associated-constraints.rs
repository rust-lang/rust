#![feature(negative_bounds, associated_type_bounds)]
//~^ WARN the feature `negative_bounds` is incomplete and may not be safe to use and/or cause compiler crashes

trait Trait {
    type Assoc;
}

fn test<T: !Trait<Assoc = i32>>() {}
//~^ ERROR associated type constraints not allowed on negative bounds

fn test2<T>() where T: !Trait<Assoc = i32> {}
//~^ ERROR associated type constraints not allowed on negative bounds

fn test3<T: !Trait<Assoc: Send>>() {}
//~^ ERROR associated type constraints not allowed on negative bounds

fn test4<T>() where T: !Trait<Assoc: Send> {}
//~^ ERROR associated type constraints not allowed on negative bounds

fn main() {}
