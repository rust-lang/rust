pub struct Struct<T> {
    pub p: T,
}

impl<T> Struct<T> {
    pub fn method(&self) {}

    pub fn some_mutable_method(&mut self) {}
}

thread_local! {
    static STRUCT: Struct<u32> = Struct {
        p: 42_u32
    };
}

fn main() {
    STRUCT.method();
    //~^ ERROR no method named `method` found for struct `LocalKey<T>` in the current scope [E0599]
    //~| HELP use `with` or `try_with` to access thread local storage

    let item = std::mem::MaybeUninit::new(Struct { p: 42_u32 });
    item.method();
    //~^ ERROR no method named `method` found for union `MaybeUninit<T>` in the current scope [E0599]
    //~| HELP if this `MaybeUninit<Struct<u32>>` has been initialized, use one of the `assume_init` methods to access the inner value
}
