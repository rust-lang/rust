#![feature(type_alias_impl_trait)]
// check-pass

fn main() {
    let x = || {
        type Tait = impl Sized;
        let y: Tait = ();
    };

    let y = || {
        type Tait = impl std::fmt::Debug;
        let y: Tait = ();
        y
    };
    let mut z = y();
    z = ();
}
