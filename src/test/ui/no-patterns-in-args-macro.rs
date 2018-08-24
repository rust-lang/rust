macro_rules! m {
    ($pat: pat) => {
        trait Tr {
            fn trait_method($pat: u8);
        }

        type A = fn($pat: u8);

        extern {
            fn foreign_fn($pat: u8);
        }
    }
}

mod good_pat {
    m!(good_pat); // OK
}

mod bad_pat {
    m!((bad, pat));
    //~^ ERROR patterns aren't allowed in function pointer types
    //~| ERROR patterns aren't allowed in foreign function declarations
    //~| ERROR patterns aren't allowed in methods without bodies
}

fn main() {}
