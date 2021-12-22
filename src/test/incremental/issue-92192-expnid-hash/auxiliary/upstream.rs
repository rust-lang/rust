macro_rules! make_struct {
    () => {
        pub struct Foo;
    }
}


#[cfg(rpass1)]
make_struct!();

#[cfg(rpass2)]
make_struct!();
