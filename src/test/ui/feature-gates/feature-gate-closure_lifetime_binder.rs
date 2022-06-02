fn main() {
    for<> || {};
    //~^ ERROR `for<...>` binders for closures are experimental
    //~^^ ERROR `for<...>` binders for closures are not yet supported
    for<'a> || {};
    //~^ ERROR `for<...>` binders for closures are experimental
    //~^^ ERROR `for<...>` binders for closures are not yet supported
    for<'a, 'b> |_: &'a ()| {};
    //~^ ERROR `for<...>` binders for closures are experimental
    //~^^ ERROR `for<...>` binders for closures are not yet supported
    //~^^^ ERROR use of undeclared lifetime name `'a`
}
