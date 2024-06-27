fn f<T: ?for<'a> Sized>() {}
//~^ ERROR expected a trait, found type

fn main() {}
