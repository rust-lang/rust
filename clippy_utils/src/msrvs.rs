use rustc_semver::RustcVersion;

macro_rules! msrv_aliases {
    ($($major:literal,$minor:literal,$patch:literal {
        $($name:ident),* $(,)?
    })*) => {
        $($(
        pub const $name: RustcVersion = RustcVersion::new($major, $minor, $patch);
        )*)*
    };
}

// names may refer to stabilized feature flags or library items
msrv_aliases! {
    1,53,0 { OR_PATTERNS }
    1,52,0 { STR_SPLIT_ONCE }
    1,50,0 { BOOL_THEN }
    1,47,0 { TAU }
    1,46,0 { CONST_IF_MATCH }
    1,45,0 { STR_STRIP_PREFIX }
    1,43,0 { LOG2_10, LOG10_2 }
    1,42,0 { MATCHES_MACRO }
    1,41,0 { RE_REBALANCING_COHERENCE, RESULT_MAP_OR_ELSE }
    1,40,0 { MEM_TAKE, NON_EXHAUSTIVE, OPTION_AS_DEREF }
    1,38,0 { POINTER_CAST }
    1,37,0 { TYPE_ALIAS_ENUM_VARIANTS }
    1,36,0 { ITERATOR_COPIED }
    1,35,0 { OPTION_COPIED, RANGE_CONTAINS }
    1,34,0 { TRY_FROM }
    1,30,0 { ITERATOR_FIND_MAP }
    1,17,0 { FIELD_INIT_SHORTHAND, STATIC_IN_CONST }
    1,16,0 { STR_REPEAT }
}
