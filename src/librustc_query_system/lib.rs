#![feature(const_fn)]
#![feature(const_if_match)]
#![feature(const_panic)]
#![feature(core_intrinsics)]
#![feature(specialization)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate log;

pub mod dep_graph;

pub trait HashStableContext {
    fn debug_dep_tasks(&self) -> bool;
}

/// Something that can provide a stable hashing context.
pub trait HashStableContextProvider<Ctxt> {
    fn get_stable_hashing_context(&self) -> Ctxt;
}

impl<Ctxt, T: HashStableContextProvider<Ctxt>> HashStableContextProvider<Ctxt> for &T {
    fn get_stable_hashing_context(&self) -> Ctxt {
        (**self).get_stable_hashing_context()
    }
}

impl<Ctxt, T: HashStableContextProvider<Ctxt>> HashStableContextProvider<Ctxt> for &mut T {
    fn get_stable_hashing_context(&self) -> Ctxt {
        (**self).get_stable_hashing_context()
    }
}
