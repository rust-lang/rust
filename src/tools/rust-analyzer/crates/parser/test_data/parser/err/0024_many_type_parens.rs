fn f<T: (Copy) + (?Sized) + (for<'a> Trait<'a>)>() {}

fn main() {
    let _: Box<(Copy) + (?Sized) + (for<'a> Trait<'a>)>;
    let _: Box<(?Sized) + (for<'a> Trait<'a>) + (Copy)>;
    let _: Box<(for<'a> Trait<'a>) + (Copy) + (?Sized)>;
}
