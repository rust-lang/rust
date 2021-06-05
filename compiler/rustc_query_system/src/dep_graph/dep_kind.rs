use super::DepNode;

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        #[macro_export]
        macro_rules! make_dep_kind_array {
            ($mod:ident) => {[ $($mod::$variant()),* ]};
        }

        /// This enum serves as an index into arrays built by `make_dep_kind_array`.
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
