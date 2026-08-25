fn bug<T>() -> impl CallbackMarker< Item = [(); { |_: &mut ()| 3; 4 }] > {}
//~^ ERROR cannot find trait `CallbackMarker` in this scope

fn main() {}
