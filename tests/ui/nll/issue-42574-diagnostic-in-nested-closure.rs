// This test illustrates a case where full NLL (enabled by the feature
// switch below) produces superior diagnostics to the NLL-migrate
// mode.

fn doit(data: &'static mut ()) {
    || doit(data);
    //~^ ERROR lifetime may not live long enough
    //~| ERROR `data` does not live long enough
}

fn main() { }
