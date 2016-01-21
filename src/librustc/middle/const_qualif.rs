// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Const qualification, from partial to completely promotable.
bitflags! {
    #[derive(RustcEncodable, RustcDecodable)]
    flags ConstQualif: u8 {
        // Inner mutability (can not be placed behind a reference) or behind
        // &mut in a non-global expression. Can be copied from static memory.
        const MUTABLE_MEM        = 1 << 0,
        // Constant value with a type that implements Drop. Can be copied
        // from static memory, similar to MUTABLE_MEM.
        const NEEDS_DROP         = 1 << 1,
        // Even if the value can be placed in static memory, copying it from
        // there is more expensive than in-place instantiation, and/or it may
        // be too large. This applies to [T; N] and everything containing it.
        // N.B.: references need to clear this flag to not end up on the stack.
        const PREFER_IN_PLACE    = 1 << 2,
        // May use more than 0 bytes of memory, doesn't impact the constness
        // directly, but is not allowed to be borrowed mutably in a constant.
        const NON_ZERO_SIZED     = 1 << 3,
        // Actually borrowed, has to always be in static memory. Does not
        // propagate, and requires the expression to behave like a 'static
        // lvalue. The set of expressions with this flag is the minimum
        // that have to be promoted.
        const HAS_STATIC_BORROWS = 1 << 4,
        // Invalid const for miscellaneous reasons (e.g. not implemented).
        const NOT_CONST          = 1 << 5,

        // Borrowing the expression won't produce &'static T if any of these
        // bits are set, though the value could be copied from static memory
        // if `NOT_CONST` isn't set.
        const NON_STATIC_BORROWS = ConstQualif::MUTABLE_MEM.bits |
                                   ConstQualif::NEEDS_DROP.bits |
                                   ConstQualif::NOT_CONST.bits
    }
}
