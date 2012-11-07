// This test is brittle!
// xfail-pretty - the pretty tests lose path information, breaking include!
#[legacy_exports];

mod m1 {
    #[legacy_exports];
    mod m2 {
        #[legacy_exports];
        fn where_am_i() -> ~str { module_path!() }
    }
}

fn main() {
    assert(line!() == 14);
    assert(col!() == 11);
    assert(file!().ends_with(~"syntax-extension-source-utils.rs"));
    assert(stringify!((2*3) + 5) == ~"(2 * 3) + 5");
    assert(include!("syntax-extension-source-utils-files/includeme.fragment")
           == ~"victory robot 6");

    assert(
        include_str!("syntax-extension-source-utils-files/includeme.fragment")
        .starts_with(~"/* this is for "));
    assert(
        include_bin!("syntax-extension-source-utils-files/includeme.fragment")
        [1] == (42 as u8)); // '*'
    // The Windows tests are wrapped in an extra module for some reason
    assert(m1::m2::where_am_i().ends_with(~"m1::m2"));
}
