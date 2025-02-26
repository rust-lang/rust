// check that panics in destructors during assignment do not leave
// destroyed values lying around for other destructors to observe.

//@ run-fail
//@ error-pattern:panicking destructors ftw!
//@ needs-subprocess

struct Observer<'a>(&'a mut FilledOnDrop);

struct FilledOnDrop(u32);
impl Drop for FilledOnDrop {
    fn drop(&mut self) {
        if self.0 == 0 {
            // this is only set during the destructor - safe
            // code should not be able to observe this.
            self.0 = 0x1c1c1c1c;
            panic!("panicking destructors ftw!");
        }
    }
}

impl<'a> Drop for Observer<'a> {
    fn drop(&mut self) {
        assert_eq!(self.0 .0, 1);
    }
}

fn foo(b: &mut Observer) {
    *b.0 = FilledOnDrop(1);
}

fn main() {
    let mut bomb = FilledOnDrop(0);
    let mut observer = Observer(&mut bomb);
    foo(&mut observer);
}
