// This test checks the combination of well known names, their activation via names(), the usage of
// partial values() with a --cfg and test that we also correctly lint on the `cfg!` macro and
// `cfg_attr` attribute.
//
// check-pass
// compile-flags: --check-cfg=names() --check-cfg=values(feature,"foo") --cfg feature="bar" -Z unstable-options

#[cfg(windows)]
fn do_windows_stuff() {}

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
fn do_windows_stuff() {}

#[cfg(feature = "foo")]
fn use_foo() {}

#[cfg(feature = "bar")]
fn use_bar() {}

#[cfg(feature = "zebra")]
//~^ WARNING unexpected `cfg` condition value
fn use_zebra() {}

#[cfg_attr(uu, test)]
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
}

fn main() {}
