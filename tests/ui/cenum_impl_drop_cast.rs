#![deny(cenum_impl_drop_cast)]

enum E {
    A = 0,
}

impl Drop for E {
    fn drop(&mut self) {
        println!("Drop");
    }
}

fn main() {
    let e = E::A;
    let i = e as u32;
    //~^ ERROR cannot cast enum `E` into integer `u32` because it implements `Drop`
    //~| WARN this was previously accepted
}
