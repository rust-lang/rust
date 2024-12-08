#![allow(unexpected_cfgs)] // since we want to recognize them as unexpected

pub mod inner {
    #[cfg(FALSE)] //~ NOTE the item is gated here
    pub fn uwu() {}
    //~^ NOTE found an item that was configured out

    #[cfg(FALSE)] //~ NOTE the item is gated here
    //~^ NOTE the item is gated here
    //~| NOTE the item is gated here
    pub mod doesnt_exist {
        //~^ NOTE found an item that was configured out
        //~| NOTE found an item that was configured out
        //~| NOTE found an item that was configured out
        pub fn hello() {}
        pub mod hi {}
    }

    pub mod wrong {
        #[cfg(feature = "suggesting me fails the test!!")]
        pub fn meow() {}
    }

    pub mod right {
        #[cfg(feature = "what-a-cool-feature")] //~ NOTE the item is gated behind the `what-a-cool-feature` feature
        pub fn meow() {}
        //~^ NOTE found an item that was configured out
    }
}

mod placeholder {
    use super::inner::doesnt_exist;
    //~^ ERROR unresolved import `super::inner::doesnt_exist`
    //~| NOTE no `doesnt_exist` in `inner`
    use super::inner::doesnt_exist::hi;
    //~^ ERROR unresolved import `super::inner::doesnt_exist`
    //~| NOTE could not find `doesnt_exist` in `inner`
}

#[cfg(i_dont_exist_and_you_can_do_nothing_about_it)]
pub fn vanished() {}

fn main() {
    // There is no uwu at this path - no diagnostic.
    uwu(); //~ ERROR cannot find function
    //~^ NOTE not found in this scope

    // It does exist here - diagnostic.
    inner::uwu(); //~ ERROR cannot find function
    //~| NOTE not found in `inner`

    // The module isn't found - we would like to get a diagnostic, but currently don't due to
    // the awkward way the resolver diagnostics are currently implemented.
    inner::doesnt_exist::hello(); //~ ERROR failed to resolve
    //~| NOTE could not find `doesnt_exist` in `inner`

    // It should find the one in the right module, not the wrong one.
    inner::right::meow(); //~ ERROR cannot find function
    //~| NOTE not found in `inner::right

    // Exists in the crate root - we would generally want a diagnostic,
    // but currently don't have one.
    // Not that it matters much though, this is highly unlikely to confuse anyone.
    vanished(); //~ ERROR cannot find function
    //~^ NOTE not found in this scope
}
