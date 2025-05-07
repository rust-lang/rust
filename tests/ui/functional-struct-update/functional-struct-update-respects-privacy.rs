// RFC 736 (and Issue 21407): functional struct update should respect privacy.

// The `foo` module attempts to maintains an invariant that each `S`
// has a unique `u64` id.

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use self::foo::S;
mod foo {
    use std::cell::{UnsafeCell};

    static mut COUNT : UnsafeCell<u64> = UnsafeCell::new(1);

    pub struct S { pub a: u8, pub b: String, secret_uid: u64 }

    pub fn make_secrets(a: u8, b: String) -> S {
        let val = unsafe { let p = COUNT.get(); let val = *p; *p = val + 1; val };
        println!("creating {}, uid {}", b, val);
        S { a: a, b: b, secret_uid: val }
    }

    impl Drop for S {
        fn drop(&mut self) {
            println!("dropping {}, uid {}", self.b, self.secret_uid);
        }
    }
}

fn main() {
    let s_1 = foo::make_secrets(3, format!("ess one"));
    let s_2 = foo::S { b: format!("ess two"), ..s_1 }; // FRU ...
    //~^ ERROR field `secret_uid` of struct `S` is private
    println!("main forged an S named: {}", s_2.b);
    // at end of scope, ... both s_1 *and* s_2 get dropped.  Boom!
}
