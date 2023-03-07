// build-pass (FIXME(62277): could be check-pass?)

// Check that method probing ObjectCandidate works in the presence of
// auto traits and/or HRTBs.

mod internal {
    pub trait MyObject<'a> {
        type Output;

        fn foo(&self) -> Self::Output;
    }

    impl<'a> MyObject<'a> for () {
        type Output = &'a u32;

        fn foo(&self) -> Self::Output { &4 }
    }
}

fn t1(d: &dyn for<'a> internal::MyObject<'a, Output=&'a u32>) {
    d.foo();
}

fn t2(d: &dyn internal::MyObject<'static, Output=&'static u32>) {
    d.foo();
}

fn t3(d: &(dyn for<'a> internal::MyObject<'a, Output=&'a u32> + Sync)) {
    d.foo();
}

fn t4(d: &(dyn internal::MyObject<'static, Output=&'static u32> + Sync)) {
    d.foo();
}

fn main() {
    t1(&());
    t2(&());
    t3(&());
    t4(&());
}
