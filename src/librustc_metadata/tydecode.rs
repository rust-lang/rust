// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Type decoding

// tjc note: Would be great to have a `match check` macro equivalent
// for some of these

#![allow(non_camel_case_types)]

use rustc::hir;

use rustc::hir::def_id::{DefId, DefIndex};
use middle::region;
use rustc::ty::subst::{Kind, Substs};
use rustc::ty::{self, ToPredicate, Ty, TyCtxt, TypeFoldable};

use rbml;
use rbml::leb128;
use std::str;
use syntax::abi;
use syntax::ast;
use syntax::parse::token;

// Compact string representation for Ty values. API TyStr &
// parse_from_str. Extra parameters are for converting to/from def_ids in the
// data buffer. Whatever format you choose should not contain pipe characters.

pub type DefIdConvert<'a> = &'a mut FnMut(DefId) -> DefId;

pub struct TyDecoder<'a, 'tcx: 'a> {
    data: &'a [u8],
    krate: ast::CrateNum,
    pos: usize,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    conv_def_id: DefIdConvert<'a>,
}

impl<'a,'tcx> TyDecoder<'a,'tcx> {
    pub fn with_doc(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    crate_num: ast::CrateNum,
                    doc: rbml::Doc<'a>,
                    conv: DefIdConvert<'a>)
                    -> TyDecoder<'a,'tcx> {
        TyDecoder::new(doc.data, crate_num, doc.start, tcx, conv)
    }

    pub fn new(data: &'a [u8],
               crate_num: ast::CrateNum,
               pos: usize,
               tcx: TyCtxt<'a, 'tcx, 'tcx>,
               conv: DefIdConvert<'a>)
               -> TyDecoder<'a, 'tcx> {
        TyDecoder {
            data: data,
            krate: crate_num,
            pos: pos,
            tcx: tcx,
            conv_def_id: conv,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn peek(&self) -> char {
        self.data[self.pos] as char
    }

    fn next(&mut self) -> char {
        let ch = self.data[self.pos] as char;
        self.pos = self.pos + 1;
        return ch;
    }

    fn next_byte(&mut self) -> u8 {
        let b = self.data[self.pos];
        self.pos = self.pos + 1;
        return b;
    }

    fn scan<F>(&mut self, mut is_last: F) -> &'a [u8]
        where F: FnMut(char) -> bool,
    {
        let start_pos = self.pos;
        debug!("scan: '{}' (start)", self.data[self.pos] as char);
        while !is_last(self.data[self.pos] as char) {
            self.pos += 1;
            debug!("scan: '{}'", self.data[self.pos] as char);
        }
        let end_pos = self.pos;
        self.pos += 1;
        return &self.data[start_pos..end_pos];
    }

    fn parse_vuint(&mut self) -> usize {
        let (value, bytes_read) = leb128::read_unsigned_leb128(self.data,
                                                               self.pos);
        self.pos += bytes_read;
        value as usize
    }

    fn parse_name(&mut self, last: char) -> ast::Name {
        fn is_last(b: char, c: char) -> bool { return c == b; }
        let bytes = self.scan(|a| is_last(last, a));
        token::intern(str::from_utf8(bytes).unwrap())
    }

    fn parse_size(&mut self) -> Option<usize> {
        assert_eq!(self.next(), '/');

        if self.peek() == '|' {
            assert_eq!(self.next(), '|');
            None
        } else {
            let n = self.parse_uint();
            assert_eq!(self.next(), '|');
            Some(n)
        }
    }

    pub fn parse_substs(&mut self) -> &'tcx Substs<'tcx> {
        let mut params = vec![];
        assert_eq!(self.next(), '[');
        while self.peek() != ']' {
            let k = match self.next() {
                'r' => Kind::from(self.parse_region()),
                't' => Kind::from(self.parse_ty()),
                _ => bug!()
            };
            params.push(k);
        }
        assert_eq!(self.next(), ']');

        Substs::new(self.tcx, params)
    }

    pub fn parse_generics(&mut self) -> &'tcx ty::Generics<'tcx> {
        let parent = self.parse_opt(|this| this.parse_def());
        let parent_regions = self.parse_u32();
        assert_eq!(self.next(), '|');
        let parent_types = self.parse_u32();

        let mut regions = vec![];
        let mut types = vec![];
        assert_eq!(self.next(), '[');
        while self.peek() != '|' {
            regions.push(self.parse_region_param_def());
        }
        assert_eq!(self.next(), '|');
        while self.peek() != ']' {
            types.push(self.parse_type_param_def());
        }
        assert_eq!(self.next(), ']');

        self.tcx.alloc_generics(ty::Generics {
            parent: parent,
            parent_regions: parent_regions,
            parent_types: parent_types,
            regions: regions,
            types: types,
            has_self: self.next() == 'S'
        })
    }

    fn parse_bound_region(&mut self) -> ty::BoundRegion {
        match self.next() {
            'a' => {
                let id = self.parse_u32();
                assert_eq!(self.next(), '|');
                ty::BrAnon(id)
            }
            '[' => {
                let def = self.parse_def();
                let name = token::intern(&self.parse_str('|'));
                let issue32330 = match self.next() {
                    'n' => {
                        assert_eq!(self.next(), ']');
                        ty::Issue32330::WontChange
                    }
                    'y' => {
                        ty::Issue32330::WillChange {
                            fn_def_id: self.parse_def(),
                            region_name: token::intern(&self.parse_str(']')),
                        }
                    }
                    c => panic!("expected n or y not {}", c)
                };
                ty::BrNamed(def, name, issue32330)
            }
            'f' => {
                let id = self.parse_u32();
                assert_eq!(self.next(), '|');
                ty::BrFresh(id)
            }
            'e' => ty::BrEnv,
            _ => bug!("parse_bound_region: bad input")
        }
    }

    pub fn parse_region(&mut self) -> &'tcx ty::Region {
        self.tcx.mk_region(match self.next() {
            'b' => {
                assert_eq!(self.next(), '[');
                let id = ty::DebruijnIndex::new(self.parse_u32());
                assert_eq!(self.next(), '|');
                let br = self.parse_bound_region();
                assert_eq!(self.next(), ']');
                ty::ReLateBound(id, br)
            }
            'B' => {
                assert_eq!(self.next(), '[');
                let index = self.parse_u32();
                assert_eq!(self.next(), '|');
                let name = token::intern(&self.parse_str(']'));
                ty::ReEarlyBound(ty::EarlyBoundRegion {
                    index: index,
                    name: name
                })
            }
            'f' => {
                assert_eq!(self.next(), '[');
                let scope = self.parse_scope();
                assert_eq!(self.next(), '|');
                let br = self.parse_bound_region();
                assert_eq!(self.next(), ']');
                ty::ReFree(ty::FreeRegion { scope: scope,
                                            bound_region: br})
            }
            's' => {
                let scope = self.parse_scope();
                assert_eq!(self.next(), '|');
                ty::ReScope(scope)
            }
            't' => ty::ReStatic,
            'e' => ty::ReEmpty,
            'E' => ty::ReErased,
            _ => bug!("parse_region: bad input")
        })
    }

    fn parse_scope(&mut self) -> region::CodeExtent {
        self.tcx.region_maps.bogus_code_extent(match self.next() {
            // This creates scopes with the wrong NodeId. This isn't
            // actually a problem because scopes only exist *within*
            // functions, and functions aren't loaded until trans which
            // doesn't care about regions.
            //
            // May still be worth fixing though.
            'C' => {
                assert_eq!(self.next(), '[');
                let fn_id = self.parse_uint() as ast::NodeId;
                assert_eq!(self.next(), '|');
                let body_id = self.parse_uint() as ast::NodeId;
                assert_eq!(self.next(), ']');
                region::CodeExtentData::CallSiteScope {
                    fn_id: fn_id, body_id: body_id
                }
            }
            // This creates scopes with the wrong NodeId. (See note above.)
            'P' => {
                assert_eq!(self.next(), '[');
                let fn_id = self.parse_uint() as ast::NodeId;
                assert_eq!(self.next(), '|');
                let body_id = self.parse_uint() as ast::NodeId;
                assert_eq!(self.next(), ']');
                region::CodeExtentData::ParameterScope {
                    fn_id: fn_id, body_id: body_id
                }
            }
            'M' => {
                let node_id = self.parse_uint() as ast::NodeId;
                region::CodeExtentData::Misc(node_id)
            }
            'D' => {
                let node_id = self.parse_uint() as ast::NodeId;
                region::CodeExtentData::DestructionScope(node_id)
            }
            'B' => {
                assert_eq!(self.next(), '[');
                let node_id = self.parse_uint() as ast::NodeId;
                assert_eq!(self.next(), '|');
                let first_stmt_index = self.parse_u32();
                assert_eq!(self.next(), ']');
                let block_remainder = region::BlockRemainder {
                    block: node_id, first_statement_index: first_stmt_index,
                };
                region::CodeExtentData::Remainder(block_remainder)
            }
            _ => bug!("parse_scope: bad input")
        })
    }

    fn parse_opt<T, F>(&mut self, f: F) -> Option<T>
        where F: FnOnce(&mut TyDecoder<'a, 'tcx>) -> T,
    {
        match self.next() {
            'n' => None,
            's' => Some(f(self)),
            _ => bug!("parse_opt: bad input")
        }
    }

    fn parse_str(&mut self, term: char) -> String {
        let mut result = String::new();
        while self.peek() != term {
            unsafe {
                result.as_mut_vec().extend_from_slice(&[self.next_byte()])
            }
        }
        self.next();
        result
    }

    pub fn parse_trait_ref(&mut self) -> ty::TraitRef<'tcx> {
        ty::TraitRef {
            def_id: self.parse_def(),
            substs: self.parse_substs()
        }
    }

    pub fn parse_existential_trait_ref(&mut self) -> ty::ExistentialTraitRef<'tcx> {
        ty::ExistentialTraitRef {
            def_id: self.parse_def(),
            substs: self.parse_substs()
        }
    }

    pub fn parse_ty(&mut self) -> Ty<'tcx> {
        let tcx = self.tcx;
        match self.next() {
            'b' => return tcx.types.bool,
            '!' => return tcx.types.never,
            'i' => { /* eat the s of is */ self.next(); return tcx.types.isize },
            'u' => { /* eat the s of us */ self.next(); return tcx.types.usize },
            'M' => {
                match self.next() {
                    'b' => return tcx.types.u8,
                    'w' => return tcx.types.u16,
                    'l' => return tcx.types.u32,
                    'd' => return tcx.types.u64,
                    'B' => return tcx.types.i8,
                    'W' => return tcx.types.i16,
                    'L' => return tcx.types.i32,
                    'D' => return tcx.types.i64,
                    'f' => return tcx.types.f32,
                    'F' => return tcx.types.f64,
                    _ => bug!("parse_ty: bad numeric type")
                }
            }
            'c' => return tcx.types.char,
            't' => {
                assert_eq!(self.next(), '[');
                let did = self.parse_def();
                let substs = self.parse_substs();
                assert_eq!(self.next(), ']');
                let def = self.tcx.lookup_adt_def(did);
                return tcx.mk_enum(def, substs);
            }
            'x' => {
                assert_eq!(self.next(), '[');
                let trait_ref = ty::Binder(self.parse_existential_trait_ref());
                let builtin_bounds = self.parse_builtin_bounds();
                let region_bound = self.parse_region();
                let mut projection_bounds = Vec::new();

                loop {
                    match self.next() {
                        'P' => {
                            let bound = self.parse_existential_projection();
                            projection_bounds.push(ty::Binder(bound));
                        }
                        '.' => { break; }
                        c => {
                            bug!("parse_bounds: bad bounds ('{}')", c)
                        }
                    }
                }
                assert_eq!(self.next(), ']');
                return tcx.mk_trait(ty::TraitObject {
                    principal: trait_ref,
                    region_bound: region_bound,
                    builtin_bounds: builtin_bounds,
                    projection_bounds: projection_bounds
                });
            }
            'p' => {
                assert_eq!(self.next(), '[');
                let index = self.parse_u32();
                assert_eq!(self.next(), '|');
                let name = token::intern(&self.parse_str(']'));
                return tcx.mk_param(index, name);
            }
            '~' => return tcx.mk_box(self.parse_ty()),
            '*' => return tcx.mk_ptr(self.parse_mt()),
            '&' => {
                return tcx.mk_ref(self.parse_region(), self.parse_mt());
            }
            'V' => {
                let t = self.parse_ty();
                return match self.parse_size() {
                    Some(n) => tcx.mk_array(t, n),
                    None => tcx.mk_slice(t)
                };
            }
            'v' => {
                return tcx.mk_str();
            }
            'T' => {
                assert_eq!(self.next(), '[');
                let mut params = Vec::new();
                while self.peek() != ']' { params.push(self.parse_ty()); }
                self.pos = self.pos + 1;
                return tcx.mk_tup(params);
            }
            'F' => {
                let def_id = self.parse_def();
                let substs = self.parse_substs();
                return tcx.mk_fn_def(def_id, substs, self.parse_bare_fn_ty());
            }
            'G' => {
                return tcx.mk_fn_ptr(self.parse_bare_fn_ty());
            }
            '#' => {
                // This is a hacky little caching scheme. The idea is that if we encode
                // the same type twice, the second (and third, and fourth...) time we will
                // just write `#123`, where `123` is the offset in the metadata of the
                // first appearance. Now when we are *decoding*, if we see a `#123`, we
                // can first check a cache (`tcx.rcache`) for that offset. If we find something,
                // we return it (modulo closure types, see below). But if not, then we
                // jump to offset 123 and read the type from there.

                let pos = self.parse_vuint();
                let key = ty::CReaderCacheKey { cnum: self.krate, pos: pos };
                if let Some(tt) = tcx.rcache.borrow().get(&key).cloned() {
                    // If there is a closure buried in the type some where, then we
                    // need to re-convert any def ids (see case 'k', below). That means
                    // we can't reuse the cached version.
                    if !tt.has_closure_types() {
                        return tt;
                    }
                }

                let mut substate = TyDecoder::new(self.data,
                                                  self.krate,
                                                  pos,
                                                  self.tcx,
                                                  self.conv_def_id);
                let tt = substate.parse_ty();
                tcx.rcache.borrow_mut().insert(key, tt);
                return tt;
            }
            '\"' => {
                let _ = self.parse_def();
                let inner = self.parse_ty();
                inner
            }
            'a' => {
                assert_eq!(self.next(), '[');
                let did = self.parse_def();
                let substs = self.parse_substs();
                assert_eq!(self.next(), ']');
                let def = self.tcx.lookup_adt_def(did);
                return self.tcx.mk_struct(def, substs);
            }
            'k' => {
                assert_eq!(self.next(), '[');
                let did = self.parse_def();
                let substs = self.parse_substs();
                let mut tys = vec![];
                while self.peek() != '.' {
                    tys.push(self.parse_ty());
                }
                assert_eq!(self.next(), '.');
                assert_eq!(self.next(), ']');
                return self.tcx.mk_closure(did, substs, tys);
            }
            'P' => {
                assert_eq!(self.next(), '[');
                let trait_ref = self.parse_trait_ref();
                let name = token::intern(&self.parse_str(']'));
                return tcx.mk_projection(trait_ref, name);
            }
            'A' => {
                assert_eq!(self.next(), '[');
                let def_id = self.parse_def();
                let substs = self.parse_substs();
                assert_eq!(self.next(), ']');
                return self.tcx.mk_anon(def_id, substs);
            }
            'e' => {
                return tcx.types.err;
            }
            c => { bug!("unexpected char in type string: {}", c);}
        }
    }

    fn parse_mutability(&mut self) -> hir::Mutability {
        match self.peek() {
            'm' => { self.next(); hir::MutMutable }
            _ => { hir::MutImmutable }
        }
    }

    fn parse_mt(&mut self) -> ty::TypeAndMut<'tcx> {
        let m = self.parse_mutability();
        ty::TypeAndMut { ty: self.parse_ty(), mutbl: m }
    }

    fn parse_def(&mut self) -> DefId {
        let def_id = parse_defid(self.scan(|c| c == '|'));
        return (self.conv_def_id)(def_id);
    }

    fn parse_uint(&mut self) -> usize {
        let mut n = 0;
        loop {
            let cur = self.peek();
            if cur < '0' || cur > '9' { return n; }
            self.pos = self.pos + 1;
            n *= 10;
            n += (cur as usize) - ('0' as usize);
        };
    }

    fn parse_u32(&mut self) -> u32 {
        let n = self.parse_uint();
        let m = n as u32;
        assert_eq!(m as usize, n);
        m
    }

    fn parse_abi_set(&mut self) -> abi::Abi {
        assert_eq!(self.next(), '[');
        let bytes = self.scan(|c| c == ']');
        let abi_str = str::from_utf8(bytes).unwrap();
        abi::lookup(&abi_str[..]).expect(abi_str)
    }

    pub fn parse_closure_ty(&mut self) -> ty::ClosureTy<'tcx> {
        let unsafety = parse_unsafety(self.next());
        let sig = self.parse_sig();
        let abi = self.parse_abi_set();
        ty::ClosureTy {
            unsafety: unsafety,
            sig: sig,
            abi: abi,
        }
    }

    pub fn parse_bare_fn_ty(&mut self) -> &'tcx ty::BareFnTy<'tcx> {
        let unsafety = parse_unsafety(self.next());
        let abi = self.parse_abi_set();
        let sig = self.parse_sig();
        self.tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: unsafety,
            abi: abi,
            sig: sig
        })
    }

    fn parse_sig(&mut self) -> ty::PolyFnSig<'tcx> {
        assert_eq!(self.next(), '[');
        let mut inputs = Vec::new();
        while self.peek() != ']' {
            inputs.push(self.parse_ty());
        }
        self.pos += 1; // eat the ']'
        let variadic = match self.next() {
            'V' => true,
            'N' => false,
            r => bug!("bad variadic: {}", r),
        };
        let output = self.parse_ty();
        ty::Binder(ty::FnSig {inputs: inputs,
                              output: output,
                              variadic: variadic})
    }

    pub fn parse_predicate(&mut self) -> ty::Predicate<'tcx> {
        match self.next() {
            't' => ty::Binder(self.parse_trait_ref()).to_predicate(),
            'e' => ty::Binder(ty::EquatePredicate(self.parse_ty(),
                                                  self.parse_ty())).to_predicate(),
            'r' => ty::Binder(ty::OutlivesPredicate(self.parse_region(),
                                                    self.parse_region())).to_predicate(),
            'o' => ty::Binder(ty::OutlivesPredicate(self.parse_ty(),
                                                    self.parse_region())).to_predicate(),
            'p' => ty::Binder(self.parse_projection_predicate()).to_predicate(),
            'w' => ty::Predicate::WellFormed(self.parse_ty()),
            'O' => {
                let def_id = self.parse_def();
                assert_eq!(self.next(), '|');
                ty::Predicate::ObjectSafe(def_id)
            }
            'c' => {
                let def_id = self.parse_def();
                assert_eq!(self.next(), '|');
                let kind = match self.next() {
                    'f' => ty::ClosureKind::Fn,
                    'm' => ty::ClosureKind::FnMut,
                    'o' => ty::ClosureKind::FnOnce,
                    c => bug!("Encountered invalid character in metadata: {}", c)
                };
                assert_eq!(self.next(), '|');
                ty::Predicate::ClosureKind(def_id, kind)
            }
            c => bug!("Encountered invalid character in metadata: {}", c)
        }
    }

    fn parse_projection_predicate(&mut self) -> ty::ProjectionPredicate<'tcx> {
        ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                trait_ref: self.parse_trait_ref(),
                item_name: token::intern(&self.parse_str('|')),
            },
            ty: self.parse_ty(),
        }
    }

    fn parse_existential_projection(&mut self) -> ty::ExistentialProjection<'tcx> {
        ty::ExistentialProjection {
            trait_ref: self.parse_existential_trait_ref(),
            item_name: token::intern(&self.parse_str('|')),
            ty: self.parse_ty(),
        }
    }

    fn parse_type_param_def(&mut self) -> ty::TypeParameterDef<'tcx> {
        let name = self.parse_name(':');
        let def_id = self.parse_def();
        let index = self.parse_u32();
        assert_eq!(self.next(), '|');
        let default_def_id = self.parse_def();
        let default = self.parse_opt(|this| this.parse_ty());
        let object_lifetime_default = self.parse_object_lifetime_default();

        ty::TypeParameterDef {
            name: name,
            def_id: def_id,
            index: index,
            default_def_id: default_def_id,
            default: default,
            object_lifetime_default: object_lifetime_default,
        }
    }

    fn parse_region_param_def(&mut self) -> ty::RegionParameterDef<'tcx> {
        let name = self.parse_name(':');
        let def_id = self.parse_def();
        let index = self.parse_u32();
        assert_eq!(self.next(), '|');
        let mut bounds = vec![];
        loop {
            match self.next() {
                'R' => bounds.push(self.parse_region()),
                '.' => { break; }
                c => {
                    bug!("parse_region_param_def: bad bounds ('{}')", c)
                }
            }
        }
        ty::RegionParameterDef {
            name: name,
            def_id: def_id,
            index: index,
            bounds: bounds,
        }
    }


    fn parse_object_lifetime_default(&mut self) -> ty::ObjectLifetimeDefault<'tcx> {
        match self.next() {
            'a' => ty::ObjectLifetimeDefault::Ambiguous,
            'b' => ty::ObjectLifetimeDefault::BaseDefault,
            's' => {
                let region = self.parse_region();
                ty::ObjectLifetimeDefault::Specific(region)
            }
            _ => bug!("parse_object_lifetime_default: bad input")
        }
    }

    fn parse_builtin_bounds(&mut self) -> ty::BuiltinBounds {
        let mut builtin_bounds = ty::BuiltinBounds::empty();
        loop {
            match self.next() {
                'S' => {
                    builtin_bounds.insert(ty::BoundSend);
                }
                'Z' => {
                    builtin_bounds.insert(ty::BoundSized);
                }
                'P' => {
                    builtin_bounds.insert(ty::BoundCopy);
                }
                'T' => {
                    builtin_bounds.insert(ty::BoundSync);
                }
                '.' => {
                    return builtin_bounds;
                }
                c => {
                    bug!("parse_bounds: bad builtin bounds ('{}')", c)
                }
            }
        }
    }
}

// Rust metadata parsing
fn parse_defid(buf: &[u8]) -> DefId {
    let mut colon_idx = 0;
    let len = buf.len();
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        bug!();
    }

    let crate_part = &buf[0..colon_idx];
    let def_part = &buf[colon_idx + 1..len];

    let crate_num = match str::from_utf8(crate_part).ok().and_then(|s| {
        s.parse::<usize>().ok()
    }) {
        Some(cn) => cn as ast::CrateNum,
        None => bug!("internal error: parse_defid: crate number expected, found {:?}",
                       crate_part)
    };
    let def_num = match str::from_utf8(def_part).ok().and_then(|s| {
        s.parse::<usize>().ok()
    }) {
        Some(dn) => dn,
        None => bug!("internal error: parse_defid: id expected, found {:?}",
                       def_part)
    };
    let index = DefIndex::new(def_num);
    DefId { krate: crate_num, index: index }
}

fn parse_unsafety(c: char) -> hir::Unsafety {
    match c {
        'u' => hir::Unsafety::Unsafe,
        'n' => hir::Unsafety::Normal,
        _ => bug!("parse_unsafety: bad unsafety {}", c)
    }
}
