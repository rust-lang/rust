fn inner() {
    #![doc("Inner attributes allowed here")]
    //! As are ModuleDoc style comments
    {
        #![doc("Inner attributes are allowed in blocks used as statements")]
        #![doc("Being validated is not affected by duplicates")]
        //! As are ModuleDoc style comments
    };
    {
        #![doc("Inner attributes are allowed in blocks when they are the last statement of another block")]
        //! As are ModuleDoc style comments
    }
}

fn outer() {
    let _ = #[doc("Outer attributes are always allowed")] {};
}

// https://github.com/rust-lang/rust-analyzer/issues/689
impl Whatever {
    fn salsa_event(&self, event_fn: impl Fn() -> Event<Self>) {
        #![allow(unused_variables)] // this is  `inner_attr` of the block
    }
}
