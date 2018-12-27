// aux-build:static_priv_by_default.rs

extern crate static_priv_by_default;

fn foo<T>() {}

fn main() {
    // Actual public items should be public
    static_priv_by_default::a;
    static_priv_by_default::b;
    static_priv_by_default::c;
    foo::<static_priv_by_default::d>();
    foo::<static_priv_by_default::e>();

    // publicly re-exported items should be available
    static_priv_by_default::bar::e;
    static_priv_by_default::bar::f;
    static_priv_by_default::bar::g;
    foo::<static_priv_by_default::bar::h>();
    foo::<static_priv_by_default::bar::i>();

    // private items at the top should be inaccessible
    static_priv_by_default::j;
    //~^ ERROR: static `j` is private
    static_priv_by_default::k;
    //~^ ERROR: function `k` is private
    static_priv_by_default::l;
    //~^ ERROR: struct `l` is private
    foo::<static_priv_by_default::m>();
    //~^ ERROR: enum `m` is private
    foo::<static_priv_by_default::n>();
    //~^ ERROR: type alias `n` is private

    // public items in a private mod should be inaccessible
    static_priv_by_default::foo::a;
    //~^ ERROR: module `foo` is private
    static_priv_by_default::foo::b;
    //~^ ERROR: module `foo` is private
    static_priv_by_default::foo::c;
    //~^ ERROR: module `foo` is private
    foo::<static_priv_by_default::foo::d>();
    //~^ ERROR: module `foo` is private
    foo::<static_priv_by_default::foo::e>();
    //~^ ERROR: module `foo` is private
}
