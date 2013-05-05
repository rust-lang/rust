// aux-build:impl_privacy_xc_2.rs
// xfail-fast

extern mod impl_privacy_xc_2;

pub fn main() {
    let fish1 = impl_privacy_xc_2::Fish { x: 1 };
    let fish2 = impl_privacy_xc_2::Fish { x: 2 };
    io::println(if fish1.eq(&fish2) { "yes" } else { "no " });
}
