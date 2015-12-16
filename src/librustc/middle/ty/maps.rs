// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepNode, DepTrackingMapId};
use middle::def_id::DefId;
use middle::ty;
use std::marker::PhantomData;

macro_rules! dep_map_ty {
    ($name:ident : ($key:ty) -> $value:ty) => {
        pub struct $name<'tcx> {
            data: PhantomData<&'tcx ()>
        }

        impl<'tcx> DepTrackingMapId for $name<'tcx> {
            type Key = $key;
            type Value = $value;
            fn to_dep_node(key: &$key) -> DepNode { DepNode::$name(*key) }
        }
    }
}

dep_map_ty! { ImplOrTraitItems: (DefId) -> ty::ImplOrTraitItem<'tcx> }
dep_map_ty! { Tcache: (DefId) -> ty::TypeScheme<'tcx> }
