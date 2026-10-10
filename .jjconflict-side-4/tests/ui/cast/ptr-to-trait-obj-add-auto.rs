trait Trait<'a> {}

fn add_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send) {
    x as _
    //~^ ERROR cannot add auto trait `Send` to dyn bound via pointer cast
    //~| NOTE unsupported cast
    //~| NOTE this could allow UB elsewhere
    //~| HELP use `transmute` if you're sure this is sound
}

// (to test diagnostic list formatting)
fn add_multiple_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send + Sync + Unpin) {
    x as _
    //~^ ERROR cannot add auto traits `Send`, `Sync`, and `Unpin` to dyn bound via pointer cast
    //~| NOTE unsupported cast
    //~| NOTE this could allow UB elsewhere
    //~| HELP use `transmute` if you're sure this is sound
}

fn main() {}
