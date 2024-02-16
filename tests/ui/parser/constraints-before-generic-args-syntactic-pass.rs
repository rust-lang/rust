//@ check-pass

#[cfg(FALSE)]
fn syntax() {
    foo::<T = u8, T: Ord, String>();
    //~^ WARN associated type bounds are unstable
    //~| WARN unstable syntax
    foo::<T = u8, 'a, T: Ord>();
    //~^ WARN associated type bounds are unstable
    //~| WARN unstable syntax
}

fn main() {}
