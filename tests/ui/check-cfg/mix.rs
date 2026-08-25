// This test checks the combination of well known names, the usage of cfg(),
// and that no implicit cfgs is added from --cfg while also testing that
// we correctly lint on the `cfg!` macro and `cfg_attr` attribute.
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --cfg feature="bar" --cfg unknown_name
//@ compile-flags: --check-cfg=cfg(feature,values("foo"))

#[cfg(windows)]
fn do_windows_stuff() {}

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
fn do_windows_stuff() {}

#[cfg(feature)]
//~^ WARNING unexpected `cfg` condition value
fn no_feature() {}

#[cfg(feature = "foo")]
fn use_foo() {}

#[cfg(feature = "bar")]
//~^ WARNING unexpected `cfg` condition value
fn use_bar() {}

#[cfg(feature = "zebra")]
//~^ WARNING unexpected `cfg` condition value
fn use_zebra() {}

#[cfg_attr(uu, unix)]
//~^ WARNING unexpected `cfg` condition name
fn do_test() {}

#[cfg_attr(feature = "foo", no_mangle)]
fn do_test_foo() {}

fn test_cfg_macro() {
    cfg!(windows);
    cfg!(widnows);
    //~^ WARNING unexpected `cfg` condition name
    cfg!(feature = "foo");
    cfg!(feature = "bar");
    //~^ WARNING unexpected `cfg` condition value
    cfg!(feature = "zebra");
    //~^ WARNING unexpected `cfg` condition value
    cfg!(xxx = "foo");
    //~^ WARNING unexpected `cfg` condition name
    cfg!(xxx);
    //~^ WARNING unexpected `cfg` condition name
    cfg!(any(xxx, windows));
    //~^ WARNING unexpected `cfg` condition name
    cfg!(any(feature = "bad", windows));
    //~^ WARNING unexpected `cfg` condition value
    cfg!(any(windows, xxx));
    //~^ WARNING unexpected `cfg` condition name
    cfg!(all(unix, xxx));
    //~^ WARNING unexpected `cfg` condition name
    cfg!(all(aa, bb));
    //~^ WARNING unexpected `cfg` condition name
    //~| WARNING unexpected `cfg` condition name
    cfg!(any(aa, bb));
    //~^ WARNING unexpected `cfg` condition name
    //~| WARNING unexpected `cfg` condition name
    cfg!(any(unix, feature = "zebra"));
    //~^ WARNING unexpected `cfg` condition value
    cfg!(any(xxx, feature = "zebra"));
    //~^ WARNING unexpected `cfg` condition name
    //~| WARNING unexpected `cfg` condition value
    cfg!(any(xxx, unix, xxx));
    //~^ WARNING unexpected `cfg` condition name
    //~| WARNING unexpected `cfg` condition name
    cfg!(all(feature = "zebra", feature = "zebra", feature = "zebra"));
    //~^ WARNING unexpected `cfg` condition value
    //~| WARNING unexpected `cfg` condition value
    //~| WARNING unexpected `cfg` condition value
}

fn main() {}
