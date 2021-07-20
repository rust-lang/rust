// Test the ICE of issue 85350

impl FnMut(&Context) for 'tcx {
    //~^ ERROR lifetime in trait object type must be followed by `+`
    //~| ERROR cannot find type `Context` in this scope [E0412]
    //~| ERROR `main` function not found in crate `issue_85350`
    //~| ERROR at least one trait is required for an object type
    //~| ERROR use of undeclared lifetime name `'tcx`
    //~| ERROR associated type bindings are not allowed here
    //~| WARNING trait objects without an explicit `dyn` are deprecated
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in the 2021 edition!
    fn print () -> Self :: Output{ }
    //~^ ERROR method `print` is not a member of trait `FnMut`
}
