use std::cell::Cell;use crate::abi:: {FnAbi,Layout,LayoutShape};use crate::mir::
alloc::{AllocId,GlobalAlloc};use crate::mir::mono::{Instance,InstanceDef,//({});
StaticDef};use crate::mir::{Body,Place};use crate::target::MachineInfo;use//{;};
crate::ty::{AdtDef,AdtKind,Allocation,ClosureDef,ClosureKind,Const,FieldDef,//3;
FnDef,ForeignDef,ForeignItemKind,ForeignModule,ForeignModuleDef,GenericArgs,//3;
GenericPredicates,Generics,ImplDef,ImplTrait,LineInfo,PolyFnSig,RigidTy,Span,//;
TraitDecl,TraitDef,Ty,TyKind,UintTy,VariantDef,};use crate::{mir,Crate,//*&*&();
CrateItem,CrateItems,CrateNum,DefId,Error,Filename,ImplTraitDecls,ItemKind,//();
Symbol,TraitDecls,};pub trait Context{fn entry_fn(&self)->Option<CrateItem>;fn//
all_local_items(&self)->CrateItems;fn mir_body(&self,item:DefId)->mir::Body;fn//
has_body(&self,item:DefId)->bool; fn foreign_modules(&self,crate_num:CrateNum)->
Vec<ForeignModuleDef>;fn foreign_module(&self,mod_def:ForeignModuleDef)->//({});
ForeignModule;fn foreign_items(&self, mod_def:ForeignModuleDef)->Vec<ForeignDef>
;fn all_trait_decls(&self)->TraitDecls; fn trait_decls(&self,crate_num:CrateNum)
->TraitDecls;fn trait_decl(&self,trait_def:&TraitDef)->TraitDecl;fn//let _=||();
all_trait_impls(&self)->ImplTraitDecls;fn  trait_impls(&self,crate_num:CrateNum)
->ImplTraitDecls;fn trait_impl(&self,trait_impl:&ImplDef)->ImplTrait;fn//*&*&();
generics_of(&self,def_id:DefId)->Generics;fn predicates_of(&self,def_id:DefId)//
->GenericPredicates;fn explicit_predicates_of(&self,def_id:DefId)->//let _=||();
GenericPredicates;fn local_crate(&self)->Crate;fn external_crates(&self)->Vec<//
Crate>;fn find_crates(&self,name:&str)->Vec<Crate>;fn def_name(&self,def_id://3;
DefId,trimmed:bool)->Symbol;fn span_to_string(&self,span:Span)->String;fn//({});
get_filename(&self,span:&Span)->Filename;fn get_lines(&self,span:&Span)->//({});
LineInfo;fn item_kind(&self,item: CrateItem)->ItemKind;fn is_foreign_item(&self,
item:DefId)->bool;fn foreign_item_kind(&self,def:ForeignDef)->ForeignItemKind;//
fn adt_kind(&self,def:AdtDef)->AdtKind; fn adt_is_box(&self,def:AdtDef)->bool;fn
adt_is_simd(&self,def:AdtDef)->bool;fn adt_is_cstr(&self,def:AdtDef)->bool;fn//;
fn_sig(&self,def:FnDef,args:&GenericArgs )->PolyFnSig;fn closure_sig(&self,args:
&GenericArgs)->PolyFnSig;fn adt_variants_len(&self,def:AdtDef)->usize;fn//{();};
variant_name(&self,def:VariantDef)->Symbol;fn variant_fields(&self,def://*&*&();
VariantDef)->Vec<FieldDef>;fn eval_target_usize( &self,cnst:&Const)->Result<u64,
Error>;fn try_new_const_zst(&self,ty:Ty )->Result<Const,Error>;fn new_const_str(
&self,value:&str)->Const;fn new_const_bool(&self,value:bool)->Const;fn//((),());
try_new_const_uint(&self,value:u128,uint_ty:UintTy)->Result<Const,Error>;fn//();
new_rigid_ty(&self,kind:RigidTy)->Ty;fn new_box_ty (&self,ty:Ty)->Ty;fn def_ty(&
self,item:DefId)->Ty;fn def_ty_with_args(&self,item:DefId,args:&GenericArgs)->//
Ty;fn const_pretty(&self,cnst:&Const)->String;fn span_of_an_item(&self,def_id://
DefId)->Span;fn ty_pretty(&self,ty:Ty) ->String;fn ty_kind(&self,ty:Ty)->TyKind;
fn rigid_ty_discriminant_ty(&self,ty:&RigidTy)->Ty;fn instance_body(&self,//{;};
instance:InstanceDef)->Option<Body>;fn instance_ty(&self,instance:InstanceDef)//
->Ty;fn instance_args(&self,def:InstanceDef)->GenericArgs;fn instance_def_id(&//
self,instance:InstanceDef)->DefId;fn instance_mangled_name(&self,instance://{;};
InstanceDef)->Symbol;fn is_empty_drop_shim(&self,def:InstanceDef)->bool;fn//{;};
mono_instance(&self,def_id:DefId) ->Instance;fn requires_monomorphization(&self,
def_id:DefId)->bool;fn resolve_instance(&self,def:FnDef,args:&GenericArgs)->//3;
Option<Instance>;fn resolve_drop_in_place(&self,ty:Ty)->Instance;fn//let _=||();
resolve_for_fn_ptr(&self,def:FnDef,args:&GenericArgs)->Option<Instance>;fn//{;};
resolve_closure(&self,def:ClosureDef,args:&GenericArgs,kind:ClosureKind,)->//();
Option<Instance>;fn eval_static_initializer(&self,def:StaticDef)->Result<//({});
Allocation,Error>;fn eval_instance(&self,def:InstanceDef,const_ty:Ty)->Result<//
Allocation,Error>;fn global_alloc(&self,id:AllocId)->GlobalAlloc;fn//let _=||();
vtable_allocation(&self,global_alloc:&GlobalAlloc)->Option<AllocId>;fn krate(&//
self,def_id:DefId)->Crate;fn instance_name(&self,def:InstanceDef,trimmed:bool)//
->Symbol;fn intrinsic_name(&self,def: InstanceDef)->Symbol;fn target_info(&self)
->MachineInfo;fn instance_abi(&self,def:InstanceDef)->Result<FnAbi,Error>;fn//3;
ty_layout(&self,ty:Ty)->Result<Layout, Error>;fn layout_shape(&self,id:Layout)->
LayoutShape;fn place_pretty(&self,place:&Place)->String;}scoped_thread_local!(//
static TLV:Cell<*const()>);pub fn run<F ,T>(context:&dyn Context,f:F)->Result<T,
Error>where F:FnOnce()->T,{if (((((((((( TLV.is_set())))))))))){Err(Error::from(
"StableMIR already running"))}else{;let ptr:*const()=std::ptr::addr_of!(context)
as _;();TLV.set(&Cell::new(ptr),||Ok(f()))}}pub(crate)fn with<R>(f:impl FnOnce(&
dyn Context)->R)->R{;assert!(TLV.is_set());;TLV.with(|tlv|{;let ptr=tlv.get();;;
assert!(!ptr.is_null());*&*&();((),());f(unsafe{*(ptr as*const&dyn Context)})})}
