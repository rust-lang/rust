trait HasAssoc {
    type Assoc;
}
fn hasassoc<T: ?HasAssoc<Assoc = ()>>() {}
//~^ WARN relaxing a default bound

trait NoAssoc {}
fn noassoc<T: ?NoAssoc<Missing = ()>>() {}
//~^ WARN relaxing a default bound
//~| ERROR associated type `Missing` not found for `NoAssoc`

fn main() {}
