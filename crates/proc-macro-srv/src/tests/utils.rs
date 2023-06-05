//! utils used in proc-macro tests

use expect_test::Expect;
use std::str::FromStr;

use crate::{dylib, proc_macro_test_dylib_path, ProcMacroSrv};

fn parse_string(code: &str) -> Option<crate::server::TokenStream> {
    // This is a bit strange. We need to parse a string into a token stream into
    // order to create a tt::SubTree from it in fixtures. `into_subtree` is
    // implemented by all the ABIs we have so we arbitrarily choose one ABI to
    // write a `parse_string` function for and use that. The tests don't really
    // care which ABI we're using as the `into_subtree` function isn't part of
    // the ABI and shouldn't change between ABI versions.
    crate::server::TokenStream::from_str(code).ok()
}

pub fn assert_expand(macro_name: &str, ra_fixture: &str, expect: Expect) {
    assert_expand_impl(macro_name, ra_fixture, None, expect);
}

pub fn assert_expand_attr(macro_name: &str, ra_fixture: &str, attr_args: &str, expect: Expect) {
    assert_expand_impl(macro_name, ra_fixture, Some(attr_args), expect);
}

fn assert_expand_impl(macro_name: &str, input: &str, attr: Option<&str>, expect: Expect) {
    let path = proc_macro_test_dylib_path();
    let expander = dylib::Expander::new(&path).unwrap();
    let fixture = parse_string(input).unwrap();
    let attr = attr.map(|attr| parse_string(attr).unwrap().into_subtree());

    let res = expander.expand(macro_name, &fixture.into_subtree(), attr.as_ref()).unwrap();
    expect.assert_eq(&format!("{res:?}"));
}

pub(crate) fn list() -> Vec<String> {
    let dylib_path = proc_macro_test_dylib_path();
    let mut srv = ProcMacroSrv::default();
    let res = srv.list_macros(&dylib_path).unwrap();
    res.into_iter().map(|(name, kind)| format!("{name} [{kind:?}]")).collect()
}
