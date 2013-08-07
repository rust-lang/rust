// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:static_priv_by_default.rs

extern mod static_priv_by_default;

fn main() {
    // Actual public items should be public
    use static_priv_by_default::a;
    use static_priv_by_default::b;
    use static_priv_by_default::c;
    use static_priv_by_default::d;

    // publicly re-exported items should be available
    use static_priv_by_default::e;
    use static_priv_by_default::f;
    use static_priv_by_default::g;
    use static_priv_by_default::h;

    // private items at the top should be inaccessible
    use static_priv_by_default::i;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::j;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::k;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::l;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve

    // public items in a private mod should be inaccessible
    use static_priv_by_default::foo::a;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::foo::b;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::foo::c;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
    use static_priv_by_default::foo::d;
    //~^ ERROR: unresolved import
    //~^^ ERROR: failed to resolve
}
