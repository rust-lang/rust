//@ known-bug: #127804

struct Thing;

pub trait Every {
    type Assoc;
}
impl<T: ?Sized> Every for Thing {
    type Assoc = T;
}

fn foo(_: <Thing as Every>::Assoc) {}
