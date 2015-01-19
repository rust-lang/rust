// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_diagnostic! { E0001, r##"
    This error suggests that the expression arm corresponding to the noted pattern
    will never be reached as for all possible values of the expression being matched,
    one of the preceeding patterns will match.

    This means that perhaps some of the preceeding patterns are too general, this
    one is too specific or the ordering is incorrect.
"## }

register_diagnostics! {
    E0002,
    E0003,
    E0004,
    E0005,
    E0006,
    E0007,
    E0008,
    E0009,
    E0010,
    E0011,
    E0012,
    E0014,
    E0015,
    E0016,
    E0017,
    E0018,
    E0019,
    E0020,
    E0022,
    E0109,
    E0110,
    E0133,
    E0134,
    E0135,
    E0136,
    E0137,
    E0138,
    E0139,
    E0152,
    E0158,
    E0161,
    E0162,
    E0165,
    E0170,
    E0261, // use of undeclared lifetime name
    E0262, // illegal lifetime parameter name
    E0263, // lifetime name declared twice in same scope
    E0264, // unknown external lang item
    E0265, // recursive constant
    E0266, // expected item
    E0267, // thing inside of a closure
    E0268, // thing outside of a loop
    E0269, // not all control paths return a value
    E0270, // computation may converge in a function marked as diverging
    E0271, // type mismatch resolving
    E0272, // rustc_on_unimplemented attribute refers to non-existent type parameter
    E0273, // rustc_on_unimplemented must have named format arguments
    E0274, // rustc_on_unimplemented must have a value
    E0275, // overflow evaluating requirement
    E0276, // requirement appears on impl method but not on corresponding trait method
    E0277, // trait is not implemented for type
    E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
    E0281, // type implements trait but other trait is required
    E0282, // unable to infer enough type information about
    E0283, // cannot resolve type
    E0284, // cannot resolve type
    E0285, // overflow evaluation builtin bounds
    E0296, // malformed recursion limit attribute
    E0297, // refutable pattern in for loop binding
    E0298, // mismatched types between arms
    E0299, // mismatched types between arms
    E0300, // unexpanded macro
    E0301, // cannot mutable borrow in a pattern guard
    E0302, // cannot assign in a pattern guard
    E0303, // pattern bindings are not allowed after an `@`
    E0304, // expected signed integer constant
    E0305, // expected constant
    E0306, // expected positive integer for repeat count
    E0307, // expected constant integer for repeat count
    E0308,
    E0309, // thing may not live long enough
    E0310, // thing may not live long enough
    E0311, // thing may not live long enough
    E0312, // lifetime of reference outlives lifetime of borrowed content
    E0313, // lifetime of borrowed pointer outlives lifetime of captured variable
    E0314, // closure outlives stack frame
    E0315 // cannot invoke closure outside of its lifetime
}

__build_diagnostic_array! { DIAGNOSTICS }

