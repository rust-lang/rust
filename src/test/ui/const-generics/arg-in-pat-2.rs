// check-pass
enum Generic<const N: usize> {
    Variant,
}

fn main() {
    match todo!() {
        Generic::<0usize>::Variant => todo!()
    }
}
