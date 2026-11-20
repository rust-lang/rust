//@ check-pass

trait Tr {
    type Assoc;
}

impl Tr for () {
    type Assoc = ();
}

fn main() {
    let x = &() as &(dyn Tr<Assoc = ()> + Send) as &dyn Send;
}
