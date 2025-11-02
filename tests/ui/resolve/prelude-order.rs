//@ proc-macro:macro_helpers.rs
//@ compile-flags: --crate-type=lib
//@ ignore-backends: gcc

/* There are 5 preludes and 3 namespaces. Test the order in which they are resolved.
 * See https://doc.rust-lang.org/nightly/reference/names/preludes.html.
 *
 * Macros cannot be in the type or value namespace.
 * Tools and extern crates cannot be in the macro or value namespace.
 *
 * Test the following truth tables:

Type:
| ...... |  tool  | extern | macro  | lang   | libs |
|   tool |    N/A |                   mirror
| extern | extern |    N/A |             universe
|  macro |    N/A |    N/A |    N/A |
|   lang |   tool | extern |    N/A |   N/A  |
|   libs |   tool | extern |    N/A |   X    |  N/A |

Macro:
| ...... |  tool  | extern | macro  | lang   | libs |
|   tool |    N/A |                   mirror
| extern |    N/A |    N/A |             universe
|  macro |    N/A |    N/A |    N/A |
|   lang |    N/A |    N/A |  macro |   N/A  |
|   libs |    N/A |    N/A |  macro |   X    | N/A  |

Value: N/A. Only libs has items in the value namespace.

† ambiguous
X don't care (controlled namespace with no overlap)

* Types are tested with `#[name::inner]`. Macros are tested with `#[name]`.
* WARNING: I have found in testing that attribute macros give ambiguity errors in some contexts
* instead of choosing a prelude. Have not been able to replicate.
*
* There should be 7 total tests.
* See `rustc_resolve::ident::visit_scopes` for more information,
* and for a definition of "controlled namespace".
*/

#![feature(register_tool)]

/* tool prelude */
#![register_tool(type_ns)] // extern prelude. type.
#![register_tool(i8)]      // lang   prelude. type.
#![register_tool(Sync)]    // libs   prelude. type.

/* extern prelude */
extern crate macro_helpers as type_ns; // tool prelude. type.
extern crate macro_helpers as usize;   // lang prelude. type.
extern crate macro_helpers as Option;  // libs prelude. type.

/* macro_use prelude */
#[macro_use]
extern crate macro_helpers as _;

/* lang and libs implicitly in scope */

// tool/extern -> extern
#[type_ns::inner] //~ ERROR could not find `inner` in `type_ns`
fn t1() {}

// tool/lang -> tool
#[i8::inner] // ok
fn t2() {}

// tool/libs -> tool
#[Sync::not_real] // ok
fn t3() {}

// extern/lang -> extern
#[usize::inner] //~ ERROR could not find `inner` in `usize`
fn e1() {} // NOTE: testing with `-> usize` isn't valid, crates aren't considered in that scope
           // (unless they have generic arguments, for some reason.)

// extern/libs -> extern
// https://github.com/rust-lang/rust/issues/139095
fn e2() -> Option<i32> { None } //~ ERROR: expected type, found crate

// macro/libs -> macro
#[test] //~ ERROR mismatched types
fn m1() {}

// macro/lang -> macro
#[global_allocator] //~ ERROR mismatched types
fn m2() {}

// lang/libs: no items that currently overlap, in either macro or type ns.
