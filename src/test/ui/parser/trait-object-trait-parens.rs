trait Trait<'a> {}

trait Obj {}

fn f<T: (Copy) + (?Sized) + (for<'a> Trait<'a>)>() {}

fn main() {
    let _: Box<(Obj) + (?Sized) + (for<'a> Trait<'a>)>;
    //~^ ERROR `?Trait` is not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
    let _: Box<(?Sized) + (for<'a> Trait<'a>) + (Obj)>;
    //~^ ERROR `?Trait` is not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
    let _: Box<(for<'a> Trait<'a>) + (Obj) + (?Sized)>;
    //~^ ERROR `?Trait` is not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
}
