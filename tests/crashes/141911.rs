//@ known-bug: rust-lang/rust#141911
trait MyTrait {
    fn virtualize(&self);
}
struct VirtualWrapper<T>(T, T);

impl<T: 'static> MyTrait for T {
    fn virtualize(&self) {
        const { std::ptr::null::<VirtualWrapper<T>>() as *const dyn MyTrait };
    }
}

fn main() {
    0u8.virtualize();
}
