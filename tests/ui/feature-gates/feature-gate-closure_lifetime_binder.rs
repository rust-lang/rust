fn main() {
    for<> || -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
    for<'a> || -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
    for<'a, 'b> |_: &'a ()| -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
}
