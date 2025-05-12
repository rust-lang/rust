//@ run-pass

macro_rules! compiles_fine {
    (#[$at:meta]) => {
        // test that the different types of attributes work
        #[attribute]
        /// Documentation!
        #[$at]

        // check that the attributes are recognised by requiring this
        // to be removed to avoid a compile error
        #[cfg(false)]
        static MISTYPED: () = "foo";
    }
}

// item
compiles_fine!(#[foo]);

pub fn main() {
    // statement
    compiles_fine!(#[bar]);
}
