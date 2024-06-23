//@ check-pass

trait Trait<'a> {}

fn add_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send) {
    x as _
    //~^ warning: adding an auto trait `Send` to a trait object in a pointer cast may cause UB later on
    //~| warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
