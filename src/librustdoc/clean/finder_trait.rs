// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use syntax_pos::DUMMY_SP;

use super::*;

pub trait Finder<'a, 'tcx: 'a, 'rcx: 'a> {
    fn get_cx(&self) -> &DocContext<'a, 'tcx, 'rcx>;

    // This is an ugly hack, but it's the simplest way to handle synthetic impls without greatly
    // refactoring either librustdoc or librustc. In particular, allowing new DefIds to be
    // registered after the AST is constructed would require storing the defid mapping in a
    // RefCell, decreasing the performance for normal compilation for very little gain.
    //
    // Instead, we construct 'fake' def ids, which start immediately after the last DefId in
    // DefIndexAddressSpace::Low. In the Debug impl for clean::Item, we explicitly check for fake
    // def ids, as we'll end up with a panic if we use the DefId Debug impl for fake DefIds
    fn next_def_id(&self, crate_num: CrateNum) -> DefId {
        let start_def_id = {
            let next_id = if crate_num == LOCAL_CRATE {
                self.get_cx()
                    .tcx
                    .hir
                    .definitions()
                    .def_path_table()
                    .next_id(DefIndexAddressSpace::Low)
            } else {
                self.get_cx()
                    .cstore
                    .def_path_table(crate_num)
                    .next_id(DefIndexAddressSpace::Low)
            };

            DefId {
                krate: crate_num,
                index: next_id,
            }
        };

        let mut fake_ids = self.get_cx().fake_def_ids.borrow_mut();

        let def_id = fake_ids.entry(crate_num).or_insert(start_def_id).clone();
        fake_ids.insert(
            crate_num,
            DefId {
                krate: crate_num,
                index: DefIndex::from_array_index(
                    def_id.index.as_array_index() + 1,
                    def_id.index.address_space(),
                ),
            },
        );

        MAX_DEF_ID.with(|m| {
            m.borrow_mut()
                .entry(def_id.krate.clone())
                .or_insert(start_def_id);
        });

        self.get_cx().all_fake_def_ids.borrow_mut().insert(def_id);

        def_id.clone()
    }

    fn get_real_ty<F>(&self,
                      def_id: DefId,
                      def_ctor: &F,
                      real_name: &Option<Ident>,
                      generics: &ty::Generics,
    ) -> hir::Ty
    where F: Fn(DefId) -> Def {
        let path = get_path_for_type(self.get_cx().tcx, def_id, def_ctor);
        let mut segments = path.segments.into_vec();
        let last = segments.pop().expect("segments were empty");

        segments.push(hir::PathSegment::new(
            real_name.unwrap_or(last.ident),
            self.generics_to_path_params(generics.clone()),
            false,
        ));

        let new_path = hir::Path {
            span: path.span,
            def: path.def,
            segments: HirVec::from_vec(segments),
        };

        hir::Ty {
            id: ast::DUMMY_NODE_ID,
            node: hir::TyKind::Path(hir::QPath::Resolved(None, P(new_path))),
            span: DUMMY_SP,
            hir_id: hir::DUMMY_HIR_ID,
        }
    }

    fn generics_to_path_params(&self, generics: ty::Generics) -> hir::GenericArgs {
        let mut args = vec![];

        for param in generics.params.iter() {
            match param.kind {
                ty::GenericParamDefKind::Lifetime => {
                    let name = if param.name == "" {
                        hir::ParamName::Plain(keywords::StaticLifetime.ident())
                    } else {
                        hir::ParamName::Plain(ast::Ident::from_interned_str(param.name))
                    };

                    args.push(hir::GenericArg::Lifetime(hir::Lifetime {
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                        name: hir::LifetimeName::Param(name),
                    }));
                }
                ty::GenericParamDefKind::Type {..} => {
                    args.push(hir::GenericArg::Type(self.ty_param_to_ty(param.clone())));
                }
            }
        }

        hir::GenericArgs {
            args: HirVec::from_vec(args),
            bindings: HirVec::new(),
            parenthesized: false,
        }
    }

    fn ty_param_to_ty(&self, param: ty::GenericParamDef) -> hir::Ty {
        debug!("ty_param_to_ty({:?}) {:?}", param, param.def_id);
        hir::Ty {
            id: ast::DUMMY_NODE_ID,
            node: hir::TyKind::Path(hir::QPath::Resolved(
                None,
                P(hir::Path {
                    span: DUMMY_SP,
                    def: Def::TyParam(param.def_id),
                    segments: HirVec::from_vec(vec![
                        hir::PathSegment::from_ident(Ident::from_interned_str(param.name))
                    ]),
                }),
            )),
            span: DUMMY_SP,
            hir_id: hir::DUMMY_HIR_ID,
        }
    }
}
