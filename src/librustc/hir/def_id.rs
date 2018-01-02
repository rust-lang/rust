// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty;

use rustc_data_structures::indexed_vec::Idx;
use serialize;
use std::fmt;
use std::u32;

newtype_index!(CrateNum
    {
        ENCODABLE = custom
        DEBUG_FORMAT = "crate{}",

        /// Item definitions in the currently-compiled crate would have the CrateNum
        /// LOCAL_CRATE in their DefId.
        const LOCAL_CRATE = 0,

        /// Virtual crate for builtin macros
        // FIXME(jseyfried): this is also used for custom derives until proc-macro crates get
        // `CrateNum`s.
        const BUILTIN_MACROS_CRATE = u32::MAX,

        /// A CrateNum value that indicates that something is wrong.
        const INVALID_CRATE = u32::MAX - 1,

        /// A special CrateNum that we use for the tcx.rcache when decoding from
        /// the incr. comp. cache.
        const RESERVED_FOR_INCR_COMP_CACHE = u32::MAX - 2,
    });

impl CrateNum {
    pub fn new(x: usize) -> CrateNum {
        assert!(x < (u32::MAX as usize));
        CrateNum(x as u32)
    }

    pub fn from_u32(x: u32) -> CrateNum {
        CrateNum(x)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn as_def_id(&self) -> DefId { DefId { krate: *self, index: CRATE_DEF_INDEX } }
}

impl fmt::Display for CrateNum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl serialize::UseSpecializedEncodable for CrateNum {}
impl serialize::UseSpecializedDecodable for CrateNum {}

/// A DefIndex is an index into the hir-map for a crate, identifying a
/// particular definition. It should really be considered an interned
/// shorthand for a particular DefPath.
///
/// At the moment we are allocating the numerical values of DefIndexes from two
/// address spaces: DefIndexAddressSpace::Low and DefIndexAddressSpace::High.
/// This allows us to allocate the DefIndexes of all item-likes
/// (Items, TraitItems, and ImplItems) into one of these spaces and
/// consequently use a simple array for lookup tables keyed by DefIndex and
/// known to be densely populated. This is especially important for the HIR map.
///
/// Since the DefIndex is mostly treated as an opaque ID, you probably
/// don't have to care about these address spaces.

#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, Hash, Copy)]
pub struct DefIndex(u32);

/// The crate root is always assigned index 0 by the AST Map code,
/// thanks to `NodeCollector::new`.
pub const CRATE_DEF_INDEX: DefIndex = DefIndex(0);


impl fmt::Debug for DefIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "DefIndex({}:{})",
               self.address_space().index(),
               self.as_array_index())
    }
}

impl DefIndex {
    #[inline]
    pub fn address_space(&self) -> DefIndexAddressSpace {
        match self.0 & 1 {
            0 => DefIndexAddressSpace::Low,
            1 => DefIndexAddressSpace::High,
            _ => unreachable!()
        }
    }

    /// Converts this DefIndex into a zero-based array index.
    /// This index is the offset within the given DefIndexAddressSpace.
    #[inline]
    pub fn as_array_index(&self) -> usize {
        (self.0 >> 1) as usize
    }

    #[inline]
    pub fn from_array_index(i: usize, address_space: DefIndexAddressSpace) -> DefIndex {
        DefIndex::from_raw_u32(((i << 1) | (address_space as usize)) as u32)
    }

    // Proc macros from a proc-macro crate have a kind of virtual DefIndex. This
    // function maps the index of the macro within the crate (which is also the
    // index of the macro in the CrateMetadata::proc_macros array) to the
    // corresponding DefIndex.
    pub fn from_proc_macro_index(proc_macro_index: usize) -> DefIndex {
        let def_index = DefIndex::from_array_index(proc_macro_index,
                                                   DefIndexAddressSpace::High);
        assert!(def_index != CRATE_DEF_INDEX);
        def_index
    }

    // This function is the reverse of from_proc_macro_index() above.
    pub fn to_proc_macro_index(self: DefIndex) -> usize {
        self.as_array_index()
    }

    // Don't use this if you don't know about the DefIndex encoding.
    pub fn from_raw_u32(x: u32) -> DefIndex {
        DefIndex(x)
    }

    // Don't use this if you don't know about the DefIndex encoding.
    pub fn as_raw_u32(&self) -> u32 {
        self.0
    }
}

impl serialize::UseSpecializedEncodable for DefIndex {}
impl serialize::UseSpecializedDecodable for DefIndex {}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum DefIndexAddressSpace {
    Low = 0,
    High = 1,
}

impl DefIndexAddressSpace {
    #[inline]
    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// A DefId identifies a particular *definition*, by combining a crate
/// index and a def index.
#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DefId({:?}/{}:{}",
               self.krate.index(),
               self.index.address_space().index(),
               self.index.as_array_index())?;

        ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                write!(f, " ~ {}", tcx.def_path_debug_str(*self))?;
            }
            Ok(())
        })?;

        write!(f, ")")
    }
}

impl DefId {
    /// Make a local `DefId` with the given index.
    #[inline]
    pub fn local(index: DefIndex) -> DefId {
        DefId { krate: LOCAL_CRATE, index: index }
    }

    #[inline]
    pub fn is_local(self) -> bool {
        self.krate == LOCAL_CRATE
    }

    #[inline]
    pub fn to_local(self) -> LocalDefId {
        LocalDefId::from_def_id(self)
    }
}

impl serialize::UseSpecializedEncodable for DefId {}
impl serialize::UseSpecializedDecodable for DefId {}

/// A LocalDefId is equivalent to a DefId with `krate == LOCAL_CRATE`. Since
/// we encode this information in the type, we can ensure at compile time that
/// no DefIds from upstream crates get thrown into the mix. There are quite a
/// few cases where we know that only DefIds from the local crate are expected
/// and a DefId from a different crate would signify a bug somewhere. This
/// is when LocalDefId comes in handy.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LocalDefId(DefIndex);

impl LocalDefId {

    #[inline]
    pub fn from_def_id(def_id: DefId) -> LocalDefId {
        assert!(def_id.is_local());
        LocalDefId(def_id.index)
    }

    #[inline]
    pub fn to_def_id(self) -> DefId {
        DefId {
            krate: LOCAL_CRATE,
            index: self.0
        }
    }
}

impl fmt::Debug for LocalDefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_def_id().fmt(f)
    }
}

impl serialize::UseSpecializedEncodable for LocalDefId {}
impl serialize::UseSpecializedDecodable for LocalDefId {}
