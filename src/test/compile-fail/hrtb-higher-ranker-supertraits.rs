// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a trait (`Bar`) with a higher-ranked supertrait.

trait Foo<'tcx>
{
    fn foo(&'tcx self) -> &'tcx isize;
}

trait Bar<'ccx>
    : for<'tcx> Foo<'tcx>
{
    fn bar(&'ccx self) -> &'ccx isize;
}

fn want_foo_for_some_tcx<'x,F>(f: &'x F)
    where F : Foo<'x>
{
    want_foo_for_some_tcx(f);
    want_foo_for_any_tcx(f); //~ ERROR not implemented
}

fn want_foo_for_any_tcx<F>(f: &F)
    where F : for<'tcx> Foo<'tcx>
{
    want_foo_for_some_tcx(f);
    want_foo_for_any_tcx(f);
}

fn want_bar_for_some_ccx<'x,B>(b: &B)
    where B : Bar<'x>
{
    want_foo_for_some_tcx(b);
    want_foo_for_any_tcx(b);

    want_bar_for_some_ccx(b);
    want_bar_for_any_ccx(b); //~ ERROR not implemented
}

fn want_bar_for_any_ccx<B>(b: &B)
    where B : for<'ccx> Bar<'ccx>
{
    want_foo_for_some_tcx(b);
    want_foo_for_any_tcx(b);

    want_bar_for_some_ccx(b);
    want_bar_for_any_ccx(b);
}

fn main() {}
