pub struct A<T> {
    pub v: T,
}
pub struct B<T> {
    pub v: T,
}

pub mod test {
    pub struct A<T> {
        pub v: T,
    }

    impl<T> A<T> {
        pub fn foo(&self) -> isize {
            static a: isize = 5;
            return a;
        }

        pub fn bar(&self) -> isize {
            static a: isize = 6;
            return a;
        }
    }
}
