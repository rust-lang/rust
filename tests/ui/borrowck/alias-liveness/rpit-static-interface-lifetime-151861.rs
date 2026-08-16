fn foo<'r>(r: &'r str) -> impl 'static + Into<&'r str> {
    struct Wrapper(std::ptr::NonNull<str>);

    impl<'r> Into<&'r str> for Wrapper {
        fn into(self) -> &'r str {
            unsafe {
                // SAFETY: `Wrapper` becomes an `impl use<'r> + Into<&'r str>`,
                // so it cannot yield something with any lifetime other than
                // this `'r` (or a covariant shrinkage thereof)... right?
                self.0.as_ref()
            }
        }
    }

    Wrapper(r.into())
}

fn main() {
    let a;
    {
        let s = String::from("huh");
        a = foo(&s); //~ ERROR `s` does not live long enough
    }
    let _unrelated = String::from("UB!");
    let dangling: &str = a.into();
    println!("{dangling}");
}
