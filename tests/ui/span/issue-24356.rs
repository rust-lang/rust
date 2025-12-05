// Regression test for #24356

fn main() {
    {
        use std::ops::Deref;

        struct Thing(i8);

        /*
        // Correct impl
        impl Deref for Thing {
            type Target = i8;
            fn deref(&self) -> &i8 { &self.0 }
        }
        */

        // Causes ICE
        impl Deref for Thing {
            //~^ ERROR E0046
            fn deref(&self) -> i8 { self.0 }
        }

        let thing = Thing(72);

        *thing
    };
}
