use crate::dep_graph::{DepNode, TaskDeps};
use std::fmt;
use rustc_data_structures::sync::Lock;
use std::hash::Hash;
// erase!() just makes tokens go away. It's used to specify which macro argument
// is repeated (i.e., which sub-expression of the macro we are in) but don't need
// to actually use any of the arguments.
macro_rules! erase {
    ($x:tt) => {{}};
}

macro_rules! is_anon_attr {
    (anon) => {
        true
    };
    ($attr:ident) => {
        false
    };
}

macro_rules! is_eval_always_attr {
    (eval_always) => {
        true
    };
    ($attr:ident) => {
        false
    };
}

macro_rules! contains_anon_attr {
    ($($attr:ident $(($($attr_args:tt)*))* ),*) => ({$(is_anon_attr!($attr) | )* false});
}

macro_rules! contains_eval_always_attr {
    ($($attr:ident $(($($attr_args:tt)*))* ),*) => ({$(is_eval_always_attr!($attr) | )* false});
}

macro_rules! define_dep_kind_enum {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
        #[allow(non_camel_case_types)]
        pub enum DepKind {
            $($variant),*
        }

        impl DepKind {
            pub fn is_anon(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_anon_attr!($($attrs)*) }
                    )*
                }
            }

            pub fn is_eval_always(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_eval_always_attr!($($attrs)*) }
                    )*
                }
            }

            #[allow(unreachable_code)]
            pub fn has_params(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            // tuple args
                            $({
                                erase!($tuple_arg_ty);
                                return true;
                            })*

                            false
                        }
                    )*
                }
            }
        }
    )
}

rustc_dep_node_append!([define_dep_kind_enum!][ <'tcx>
    // We use this for most things when incr. comp. is turned off.
    [] Null,

    // Represents metadata from an extern crate.
    [eval_always] CrateMetadata(CrateNum),

    [anon] TraitSelect,

    [] CompileCodegenUnit(Symbol),
]);

impl DepKind {
    pub const NULL: Self = DepKind::Null;

    pub fn debug_node(node: &DepNode, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", node.kind)?;

        if !node.kind.has_params() && !node.kind.is_anon() {
            return Ok(());
        }

        // Can't use tcx here :'(
        unimplemented!()
    }

    /*fn with_deps<OP, R>(_deps: Option<&Lock<TaskDeps<Self>>>, _op: OP) -> R where
        OP: FnOnce() -> R {
        unimplemented!()
    }

    fn read_deps<OP>(_op: OP) where
        OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps<Self>>>) {
        unimplemented!()
    }

    fn can_reconstruct_query_key(&self) -> bool {
        //DepKind::can_reconstruct_query_key(self)
        unimplemented!()
    }*/
}

/// Describe the different families of dependency nodes.
pub trait DepKindExt: Copy + fmt::Debug + Eq + Ord + Hash {
    const NULL: Self;

    /// Return whether this kind always require evaluation.
    fn is_eval_always(&self) -> bool;

    /// Return whether this kind requires additional parameters to be executed.
    fn has_params(&self) -> bool;

    /// Implementation of `std::fmt::Debug` for `DepNode`.
    fn debug_node(node: &DepNode, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Execute the operation with provided dependencies.
    fn with_deps<OP, R>(deps: Option<&Lock<TaskDeps>>, op: OP) -> R
        where
            OP: FnOnce() -> R;

    /// Access dependencies from current implicit context.
    fn read_deps<OP>(op: OP)
        where
            OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps>>);

    fn can_reconstruct_query_key(&self) -> bool;
}
