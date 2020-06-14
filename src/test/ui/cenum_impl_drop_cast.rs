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
    //~^ ERROR Cast `enum` implementing `Drop` `E` to integer `u32`
    //~| WARN this was previously accepted
}
