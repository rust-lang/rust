fn f<T: ?for<'a> Sized>() {}
//~^ ERROR `for<...>` binder should be placed before trait bound modifiers

fn main() {}
