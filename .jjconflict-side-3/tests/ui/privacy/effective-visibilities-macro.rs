#![feature(rustc_attrs, decl_macro)]

pub mod ty {
    pub mod print {
        mod pretty {
            #[rustc_effective_visibility]
            pub macro with_no_queries() {}
            //~^ ERROR  Direct: pub(in crate::ty::print), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
        }

        pub use self::pretty::with_no_queries;
        // Start visiting outer modules during macro-reachable phase.
        use crate::ty;
    }

    #[rustc_effective_visibility]
    mod sty {
    //~^ ERROR Direct: pub(self), Reexported: pub(self), Reachable: pub(self), ReachableThroughImplTrait: pub(self)
        #[rustc_effective_visibility]
        pub type Placeholder = ();
        //~^ ERROR Direct: pub(in crate::ty), Reexported: pub(in crate::ty), Reachable: pub, ReachableThroughImplTrait: pub
    }
}

fn main() {}
