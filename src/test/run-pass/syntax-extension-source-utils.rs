// This test is brittle! 
// xfail-pretty - the pretty tests lose path information, breaking #include

fn main() {
    assert(#line[] == 5u);
    assert(#col[] == 12u);
    assert(#file[].ends_with("syntax-extension-source-utils.rs"));
    assert(#stringify[(2*3) + 5] == "2 * 3 + 5");
    assert(#include["syntax-extension-source-utils-files/includeme.fragment"]
           == "victory robot 6")
}
