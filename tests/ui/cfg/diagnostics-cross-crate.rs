//@ aux-build:cfged_out.rs

extern crate cfged_out;

fn main() {
    // There is no uwu at this path - no diagnostic.
    cfged_out::uwu(); //~ ERROR cannot find function
    //~^ NOTE not found in `cfged_out`

    // It does exist here - diagnostic.
    cfged_out::inner::uwu(); //~ ERROR cannot find function
    //~^ NOTE found an item that was configured out
    //~| NOTE not found in `cfged_out::inner`

    // The module isn't found - we would like to get a diagnostic, but currently don't due to
    // the awkward way the resolver diagnostics are currently implemented.
    cfged_out::inner::doesnt_exist::hello(); //~ ERROR failed to resolve
    //~^ NOTE could not find `doesnt_exist` in `inner`
    //~| NOTE found an item that was configured out

    // It should find the one in the right module, not the wrong one.
    cfged_out::inner::right::meow(); //~ ERROR cannot find function
    //~^ NOTE found an item that was configured out
    //~| NOTE not found in `cfged_out::inner::right

    // Exists in the crate root - diagnostic.
    cfged_out::vanished(); //~ ERROR cannot find function
    //~^ NOTE found an item that was configured out
    //~| NOTE not found in `cfged_out`
}
