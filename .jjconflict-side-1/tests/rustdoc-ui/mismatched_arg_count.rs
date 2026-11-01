trait Trait<'a> {
    type Assoc;
}

type Alias<'a, T> = <T as Trait<'a>>::Assoc;

fn bar<'a, T: Trait<'a>>(_: Alias<'a, 'a, T>) {}
//~^ error: type alias takes 1 lifetime argument but 2 lifetime arguments were supplied
