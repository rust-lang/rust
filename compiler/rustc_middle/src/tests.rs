// FIXME(#27438): Right now, the unit tests of `rustc_middle` don't refer to any actual functions
//                generated in `rustc_data_structures` (all references are through generic functions),
//                but statics are referenced from time to time. Due to this Windows `dllimport` bug
//                we won't actually correctly link in the statics unless we also reference a function,
//                so be sure to reference a dummy function.
#[test]
fn noop() {
    rustc_data_structures::__noop_fix_for_windows_dllimport_issue();
}
