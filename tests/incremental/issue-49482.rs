// aux-build:issue-49482-macro-def.rs
// aux-build:issue-49482-reexport.rs
// revisions: rpass1

extern crate issue_49482_reexport;

pub trait KvStorage
{
    fn get(&self);
}

impl<K> KvStorage for Box<K>
where
    K: KvStorage + ?Sized,
{
    fn get(&self) {
        (**self).get()
    }
}

impl KvStorage for u32 {
    fn get(&self) {}
}

fn main() {
    /* force issue_49482_reexport to be loaded */
    issue_49482_reexport::foo();

    Box::new(2).get();
}
