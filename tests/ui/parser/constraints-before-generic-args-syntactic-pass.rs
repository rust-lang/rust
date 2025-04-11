//@ check-pass

#[cfg(false)]
fn syntax() {
    foo::<T = u8, T: Ord, String>();
    foo::<T = u8, 'a, T: Ord>();
}

fn main() {}
