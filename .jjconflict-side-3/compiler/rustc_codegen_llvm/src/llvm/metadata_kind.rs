use libc::c_uint;

pub(crate) use self::fixed_kinds::*;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct MetadataKindId(c_uint);

macro_rules! declare_fixed_metadata_kinds {
    (
        $(
            FIXED_MD_KIND($variant:ident, $value:literal)
        )*
    ) => {
        // Use a submodule to group all declarations into one `#[expect(..)]`.
        #[expect(dead_code)]
        mod fixed_kinds {
            use super::MetadataKindId;
            $(
                #[expect(non_upper_case_globals)]
                pub(crate) const $variant: MetadataKindId = MetadataKindId($value);
            )*
        }
    };
}

// Must be kept in sync with the corresponding static assertions in `RustWrapper.cpp`.
declare_fixed_metadata_kinds! {
    FIXED_MD_KIND(MD_dbg, 0)
    FIXED_MD_KIND(MD_tbaa, 1)
    FIXED_MD_KIND(MD_prof, 2)
    FIXED_MD_KIND(MD_fpmath, 3)
    FIXED_MD_KIND(MD_range, 4)
    FIXED_MD_KIND(MD_tbaa_struct, 5)
    FIXED_MD_KIND(MD_invariant_load, 6)
    FIXED_MD_KIND(MD_alias_scope, 7)
    FIXED_MD_KIND(MD_noalias, 8)
    FIXED_MD_KIND(MD_nontemporal, 9)
    FIXED_MD_KIND(MD_mem_parallel_loop_access, 10)
    FIXED_MD_KIND(MD_nonnull, 11)
    FIXED_MD_KIND(MD_dereferenceable, 12)
    FIXED_MD_KIND(MD_dereferenceable_or_null, 13)
    FIXED_MD_KIND(MD_make_implicit, 14)
    FIXED_MD_KIND(MD_unpredictable, 15)
    FIXED_MD_KIND(MD_invariant_group, 16)
    FIXED_MD_KIND(MD_align, 17)
    FIXED_MD_KIND(MD_loop, 18)
    FIXED_MD_KIND(MD_type, 19)
    FIXED_MD_KIND(MD_section_prefix, 20)
    FIXED_MD_KIND(MD_absolute_symbol, 21)
    FIXED_MD_KIND(MD_associated, 22)
    FIXED_MD_KIND(MD_callees, 23)
    FIXED_MD_KIND(MD_irr_loop, 24)
    FIXED_MD_KIND(MD_access_group, 25)
    FIXED_MD_KIND(MD_callback, 26)
    FIXED_MD_KIND(MD_preserve_access_index, 27)
    FIXED_MD_KIND(MD_vcall_visibility, 28)
    FIXED_MD_KIND(MD_noundef, 29)
    FIXED_MD_KIND(MD_annotation, 30)
    FIXED_MD_KIND(MD_nosanitize, 31)
    FIXED_MD_KIND(MD_func_sanitize, 32)
    FIXED_MD_KIND(MD_exclude, 33)
    FIXED_MD_KIND(MD_memprof, 34)
    FIXED_MD_KIND(MD_callsite, 35)
    FIXED_MD_KIND(MD_kcfi_type, 36)
    FIXED_MD_KIND(MD_pcsections, 37)
    FIXED_MD_KIND(MD_DIAssignID, 38)
    FIXED_MD_KIND(MD_coro_outside_frame, 39)
    FIXED_MD_KIND(MD_mmra, 40)
    FIXED_MD_KIND(MD_noalias_addrspace, 41)
}
