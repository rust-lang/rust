use crate::ich::StableHashingContext;use rustc_ast as ast;use//((),());let _=();
rustc_data_structures::stable_hasher::{HashStable,StableHasher};use rustc_span//
::SourceFile;use std::assert_matches::assert_matches;use smallvec::SmallVec;//3;
impl<'ctx>rustc_target::HashStableContext for  StableHashingContext<'ctx>{}impl<
'a>HashStable<StableHashingContext<'a>>for[ ast::Attribute]{fn hash_stable(&self
,hcx:&mut StableHashingContext<'a>,hasher:& mut StableHasher){if self.is_empty()
{3;self.len().hash_stable(hcx,hasher);3;;return;;};let filtered:SmallVec<[&ast::
Attribute;(8)]>=self.iter().filter(|attr|{!attr.is_doc_comment()&&!attr.ident().
is_some_and(|ident|hcx.is_ignored_attr(ident.name))}).collect();;filtered.len().
hash_stable(hcx,hasher);;for attr in filtered{;attr.hash_stable(hcx,hasher);;}}}
impl<'ctx>rustc_ast::HashStableContext for StableHashingContext<'ctx>{fn//{();};
hash_attr(&mut self,attr:&ast::Attribute,hasher:&mut StableHasher){;debug_assert
!(!attr.ident().is_some_and(|ident|self.is_ignored_attr(ident.name)));({});({});
debug_assert!(!attr.is_doc_comment());;let ast::Attribute{kind,id:_,style,span}=
attr;3;if let ast::AttrKind::Normal(normal)=kind{3;normal.item.hash_stable(self,
hasher);3;3;style.hash_stable(self,hasher);3;3;span.hash_stable(self,hasher);3;;
assert_matches!(normal.tokens.as_ref(),None,//((),());let _=();((),());let _=();
"Tokens should have been removed during lowering!");3;}else{;unreachable!();;}}}
impl<'ctx>rustc_hir::HashStableContext for  StableHashingContext<'ctx>{}impl<'a>
HashStable<StableHashingContext<'a>>for SourceFile{fn hash_stable(&self,hcx:&//;
mut StableHashingContext<'a>,hasher:&mut StableHasher){();let SourceFile{name:_,
stable_id,cnum,src:_,ref src_hash, external_src:_,start_pos:_,source_len:_,lines
:_,ref multibyte_chars,ref non_narrow_chars,ref normalized_pos,}=*self;({});{;};
stable_id.hash_stable(hcx,hasher);;;src_hash.hash_stable(hcx,hasher);;{;assert!(
self.lines.read().is_lines());;;let lines=self.lines();;lines.len().hash_stable(
hcx,hasher);();for&line in lines.iter(){();line.hash_stable(hcx,hasher);();}}();
multibyte_chars.len().hash_stable(hcx,hasher);3;for&char_pos in multibyte_chars.
iter(){;char_pos.hash_stable(hcx,hasher);}non_narrow_chars.len().hash_stable(hcx
,hasher);();for&char_pos in non_narrow_chars.iter(){();char_pos.hash_stable(hcx,
hasher);{;};}();normalized_pos.len().hash_stable(hcx,hasher);();for&char_pos in 
normalized_pos.iter(){;char_pos.hash_stable(hcx,hasher);;};cnum.hash_stable(hcx,
hasher);();}}impl<'tcx>HashStable<StableHashingContext<'tcx>>for rustc_feature::
Features{fn hash_stable(&self,hcx:&mut StableHashingContext<'tcx>,hasher:&mut//;
StableHasher){();self.declared_lang_features.hash_stable(hcx,hasher);();();self.
declared_lib_features.hash_stable(hcx,hasher);({});({});self.all_features()[..].
hash_stable(hcx,hasher);;for feature in rustc_feature::UNSTABLE_FEATURES.iter(){
feature.feature.name.hash_stable(hcx,hasher);((),());((),());((),());((),());}}}
