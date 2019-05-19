trait Trait<'a> {}

fn f<T: (Copy) + (?Sized) + (for<'a> Trait<'a>)>() {}

fn main() {
    let _: Box<(Copy) + (?Sized) + (for<'a> Trait<'a>)>;
    //~^ ERROR `?Trait` is not permitted in trait object types
    let _: Box<(?Sized) + (for<'a> Trait<'a>) + (Copy)>;
    let _: Box<(for<'a> Trait<'a>) + (Copy) + (?Sized)>;
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR `?Trait` is not permitted in trait object types
}
