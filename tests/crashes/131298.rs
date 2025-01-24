//@ known-bug: #131298

fn dyn_hoops<T>() -> *const dyn Iterator<Item = impl Captures> {
    loop {}
}

mod typeck {
    type Opaque = impl Sized;
    fn define() -> Opaque {
        let _: Opaque = super::dyn_hoops::<u8>();
    }
}
