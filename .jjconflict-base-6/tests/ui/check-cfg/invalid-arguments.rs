// Check that invalid --check-cfg are rejected
//
//@ check-fail
//@ no-auto-check-cfg
//@ revisions: anything_else boolean_after_values
//@ revisions: string_for_name_1 string_for_name_2 multiple_any multiple_values
//@ revisions: multiple_values_any not_empty_any not_empty_values_any
//@ revisions: values_any_missing_values values_any_before_ident ident_in_values_1
//@ revisions: ident_in_values_2 unknown_meta_item_1 unknown_meta_item_2 unknown_meta_item_3
//@ revisions: mixed_values_any mixed_any any_values giberich unterminated
//@ revisions: none_not_empty cfg_none unsafe_attr
//
//@ [anything_else]compile-flags: --check-cfg=anything_else(...)
//@ [boolean_after_values]compile-flags: --check-cfg=cfg(values(),true)
//@ [string_for_name_1]compile-flags: --check-cfg=cfg("NOT_IDENT")
//@ [string_for_name_2]compile-flags: --check-cfg=cfg(foo,"NOT_IDENT",bar)
//@ [multiple_any]compile-flags: --check-cfg=cfg(any(),any())
//@ [multiple_values]compile-flags: --check-cfg=cfg(foo,values(),values())
//@ [multiple_values_any]compile-flags: --check-cfg=cfg(foo,values(any(),any()))
//@ [not_empty_any]compile-flags: --check-cfg=cfg(any(foo))
//@ [not_empty_values_any]compile-flags: --check-cfg=cfg(foo,values(any(bar)))
//@ [values_any_missing_values]compile-flags: --check-cfg=cfg(foo,any())
//@ [values_any_before_ident]compile-flags: --check-cfg=cfg(values(any()),foo)
//@ [ident_in_values_1]compile-flags: --check-cfg=cfg(foo,values(bar))
//@ [ident_in_values_2]compile-flags: --check-cfg=cfg(foo,values("bar",bar,"bar"))
//@ [unknown_meta_item_1]compile-flags: --check-cfg=abc()
//@ [unknown_meta_item_2]compile-flags: --check-cfg=cfg(foo,test())
//@ [unknown_meta_item_3]compile-flags: --check-cfg=cfg(foo,values(test()))
//@ [none_not_empty]compile-flags: --check-cfg=cfg(foo,values(none("test")))
//@ [mixed_values_any]compile-flags: --check-cfg=cfg(foo,values("bar",any()))
//@ [mixed_any]compile-flags: --check-cfg=cfg(any(),values(any()))
//@ [any_values]compile-flags: --check-cfg=cfg(any(),values())
//@ [cfg_none]compile-flags: --check-cfg=cfg(none())
//@ [giberich]compile-flags: --check-cfg=cfg(...)
//@ [unterminated]compile-flags: --check-cfg=cfg(
//@ [unsafe_attr]compile-flags: --check-cfg=unsafe(cfg(foo))

fn main() {}

//~? ERROR invalid `--check-cfg` argument
