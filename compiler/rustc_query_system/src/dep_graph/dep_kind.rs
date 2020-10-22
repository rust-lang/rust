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
        pub enum DepKindEnum {
            $($variant),*
        }

        impl DepKindEnum {
            pub fn is_anon(&self) -> bool {
                match *self {
                    $(
                        DepKindEnum :: $variant => { contains_anon_attr!($($attrs)*) }
                    )*
                }
            }

            pub fn is_eval_always(&self) -> bool {
                match *self {
                    $(
                        DepKindEnum :: $variant => { contains_eval_always_attr!($($attrs)*) }
                    )*
                }
            }

            #[allow(unreachable_code)]
            pub fn has_params(&self) -> bool {
                match *self {
                    $(
                        DepKindEnum :: $variant => {
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
