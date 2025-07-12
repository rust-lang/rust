trait Trait<'a> {}

trait Obj {}

fn f<T: (Copy) + (?Sized) + (for<'a> Trait<'a>)>() {}

fn main() {
    let _: Box<(Obj) + (?Sized) + (for<'a> Trait<'a>)>;
    //~^ ERROR relaxed bounds are not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let _: Box<?Sized + (for<'a> Trait<'a>) + (Obj)>;
    //~^ ERROR relaxed bounds are not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let _: Box<for<'a> Trait<'a> + (Obj) + (?Sized)>;
    //~^ ERROR relaxed bounds are not permitted in trait object types
    //~| ERROR only auto traits can be used as additional traits
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
