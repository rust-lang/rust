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

    // The module isn't found - we mention that `cfgd_out` is `cfg`d out
    cfged_out::inner::cfgd_out::hello(); //~ ERROR cannot find
    //~^ NOTE could not find `cfgd_out` in `inner`
    //~| NOTE found an item that was configured out

    // The module isn't found - we mention that `selected_out` is `cfg_select`d out
    cfged_out::inner::selected_out::hello(); //~ ERROR cannot find
    //~^ NOTE could not find `selected_out` in `inner`
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
