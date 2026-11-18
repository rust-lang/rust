// Checks that a lint-check attribute applied to a foreign function parameter,
// only applies to that parameter

#![deny(dead_code)]

unsafe extern "C" {
    //~v ERROR function `unused_foreign` is never used
    fn unused_foreign(
        #[allow(dead_code)] arg: i32,
    );
}

fn main() {}
