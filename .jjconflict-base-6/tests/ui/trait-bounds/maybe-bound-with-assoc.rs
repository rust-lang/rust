trait HasAssoc {
    type Assoc;
}
fn hasassoc<T: ?HasAssoc<Assoc = ()>>() {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`

trait NoAssoc {}
fn noassoc<T: ?NoAssoc<Missing = ()>>() {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR associated type `Missing` not found for `NoAssoc`

fn main() {}
