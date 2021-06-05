use super::DepNode;

/// This struct stores metadata about each DepKind.
///
/// Information is retrieved by indexing the `DEP_KINDS` array using the integer value
/// of the `DepKind`. Overall, this allows to implement `DepContext` using this manual
/// jump table instead of large matches.
pub struct DepKindStruct {
    /// Whether the DepNode has parameters (query keys).
    pub has_params: bool,

    /// Anonymous queries cannot be replayed from one compiler invocation to the next.
    /// When their result is needed, it is recomputed. They are useful for fine-grained
    /// dependency tracking, and caching within one compiler invocation.
    pub is_anon: bool,

    /// Eval-always queries do not track their dependencies, and are always recomputed, even if
    /// their inputs have not changed since the last compiler invocation. The result is still
    /// cached within one compiler invocation.
    pub is_eval_always: bool,
}

impl std::ops::Deref for DepKind {
    type Target = DepKindStruct;
    fn deref(&self) -> &DepKindStruct {
        &DEP_KINDS[*self as usize]
    }
}

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

#[allow(non_upper_case_globals)]
pub mod dep_kind {
    use super::*;

    // We use this for most things when incr. comp. is turned off.
    pub const Null: DepKindStruct =
        DepKindStruct { has_params: false, is_anon: false, is_eval_always: false };

    pub const TraitSelect: DepKindStruct =
        DepKindStruct { has_params: false, is_anon: true, is_eval_always: false };

    pub const CompileCodegenUnit: DepKindStruct =
        DepKindStruct { has_params: true, is_anon: false, is_eval_always: false };

    pub const CompileMonoItem: DepKindStruct =
        DepKindStruct { has_params: true, is_anon: false, is_eval_always: false };

    macro_rules! define_query_dep_kinds {
        ($(
            [$($attrs:tt)*]
            $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
        ,)*) => (
            $(pub const $variant: DepKindStruct = {
                const has_params: bool = $({ erase!($tuple_arg_ty); true } |)* false;
                const is_anon: bool = contains_anon_attr!($($attrs)*);
                const is_eval_always: bool = contains_eval_always_attr!($($attrs)*);

                DepKindStruct {
                    has_params,
                    is_anon,
                    is_eval_always,
                }
            };)*
        );
    }

    rustc_dep_node_append!([define_query_dep_kinds!][]);
}

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        #[macro_export]
        macro_rules! make_dep_kind_array {
            ($mod:ident) => {[ $(($mod::$variant),)* ]};
        }

        static DEP_KINDS: &[DepKindStruct] = &make_dep_kind_array!(dep_kind);

        /// This enum serves as an index into the `DEP_KINDS` array.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
        #[allow(non_camel_case_types)]
        pub enum DepKind {
            $($variant),*
        }

        pub fn dep_kind_from_label_string(label: &str) -> Result<DepKind, ()> {
            match label {
                $(stringify!($variant) => Ok(DepKind::$variant),)*
                _ => Err(()),
            }
        }

        /// Contains variant => str representations for constructing
        /// DepNode groups for tests.
        #[allow(dead_code, non_upper_case_globals)]
        pub mod label_strs {
           $(
                pub const $variant: &str = stringify!($variant);
            )*
        }
    );
}

rustc_dep_node_append!([define_dep_nodes!][ <'tcx>
    // We use this for most things when incr. comp. is turned off.
    [] Null,

    [anon] TraitSelect,

    // WARNING: if `Symbol` is changed, make sure you update `make_compile_codegen_unit` below.
    [] CompileCodegenUnit(Symbol),

    // WARNING: if `MonoItem` is changed, make sure you update `make_compile_mono_item` below.
    // Only used by rustc_codegen_cranelift
    [] CompileMonoItem(MonoItem),
]);

// We keep a lot of `DepNode`s in memory during compilation. It's not
// required that their size stay the same, but we don't want to change
// it inadvertently. This assert just ensures we're aware of any change.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static_assert_size!(DepNode, 18);

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
static_assert_size!(DepNode, 24);

impl DepNode {
    /// Used in testing
    pub fn has_label_string(label: &str) -> bool {
        dep_kind_from_label_string(label).is_ok()
    }
}
