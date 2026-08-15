// EII implementations only accept attributes from a conservative allowlist.
// Regression test for #159015

//@ edition: 2024
//@ needs-asm-support

#![feature(coverage_attribute)]
#![feature(extern_item_impls)]
#![feature(optimize_attribute)]
#![feature(sanitize)]

#[eii]
fn allowed();

/// Sugared and explicit documentation attributes are both allowed.
#[allowed]
#[allow(dead_code)]
#[warn(unreachable_code)]
#[deny(unused_mut)]
#[forbid(unsafe_code)]
#[expect(unused_variables)]
#[cfg(all())]
#[doc = "An allowed EII implementation."]
#[cold]
#[optimize(none)]
#[coverage(off)]
#[sanitize(address = "off")]
#[must_use]
#[deprecated]
fn allowed_impl() {
    let unused = ();
}

#[eii]
fn allowed_inline();

#[allowed_inline]
#[allow(unused_attributes)]
#[cfg_attr(all(), inline)]
fn allowed_inline_impl() {}

#[eii]
fn foo();

#[foo]
#[unsafe(no_mangle)]
//~^ ERROR `#[foo]` is not allowed to have `#[no_mangle]`
fn bar() {}

#[eii]
fn baz();

#[baz]
#[unsafe(export_name = "qux")]
//~^ ERROR `#[baz]` is not allowed to have `#[export_name]`
fn qux() {}

#[eii]
fn quux();

#[quux]
#[unsafe(link_section = "__TEXT,__text")]
//~^ ERROR `#[quux]` is not allowed to have `#[link_section]`
fn corge() {}

#[eii]
fn grault();

#[grault]
#[track_caller]
//~^ ERROR `#[grault]` is not allowed to have `#[track_caller]`
fn garply() {}

#[eii]
extern "C" fn naked_attr();

#[naked_attr]
#[unsafe(naked)]
//~^ ERROR `#[naked_attr]` is not allowed to have `#[naked]`
extern "C" fn naked_attr_impl() {
    core::arch::naked_asm!("")
}

#[eii]
fn multiple_invalid_attrs();

#[multiple_invalid_attrs]
#[unsafe(no_mangle)]
//~^ ERROR `#[multiple_invalid_attrs]` is not allowed to have `#[no_mangle]`
#[track_caller]
//~^ ERROR `#[multiple_invalid_attrs]` is not allowed to have `#[track_caller]`
fn multiple_invalid_attrs_impl() {}

#[eii(static_eii)]
static STATIC_EII: u8;

#[static_eii]
#[used]
//~^ ERROR `#[static_eii]` is not allowed to have `#[used]`
static STATIC_EII_IMPL: u8 = 0;

fn main() {}
