use std::fmt::Debug;

//@ count "$.index[?(@.name=='dyn')].inner.module.items[*]" 3
//@ set sync_int_gen = "$.index[?(@.name=='SyncIntGen')].id"
//@ set ref_fn       = "$.index[?(@.name=='RefFn')].id"
//@ set weird_order  = "$.index[?(@.name=='WeirdOrder')].id"
//@ ismany "$.index[?(@.name=='dyn')].inner.module.items[*]" $sync_int_gen $ref_fn $weird_order

//@ has    "$.index[?(@.name=='SyncIntGen')].inner.type_alias"
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.generics" '{"params": [], "where_predicates": []}'
//@ has    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path"
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.path" \"Box\"
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.constraints" []
//@ count "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args" 1
//@ has    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait"
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.lifetime" \"\'static\"
//@ count "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[*]" 3
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[0].generic_params" []
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[1].generic_params" []
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[2].generic_params" []
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[0].trait.path" '"Fn"'
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[1].trait.path" '"Send"'
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[2].trait.path" '"Sync"'
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[0].trait.args" '{"parenthesized": {"inputs": [],"output": {"primitive": "i32"}}}'
pub type SyncIntGen = Box<dyn Fn() -> i32 + Send + Sync + 'static>;

//@ has "$.index[?(@.name=='RefFn')].inner.type_alias"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.generics" '{"params": [{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"}],"where_predicates": []}'
//@ has "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.is_mutable" 'false'
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.lifetime" "\"'a\""
//@ has "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.lifetime" null
//@ count "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[*]" 1
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].trait.path" '"Fn"'
//@ has "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].trait.args.parenthesized.inputs[0].borrowed_ref"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].trait.args.parenthesized.inputs[0].borrowed_ref.lifetime" "\"'b\""
//@ has "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].trait.args.parenthesized.output.borrowed_ref"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type.borrowed_ref.type.dyn_trait.traits[0].trait.args.parenthesized.output.borrowed_ref.lifetime" "\"'b\""
pub type RefFn<'a> = &'a dyn for<'b> Fn(&'b i32) -> &'b i32;

//@ is    "$.index[?(@.name=='WeirdOrder')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[0].trait.path" '"Send"'
//@ is    "$.index[?(@.name=='WeirdOrder')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.dyn_trait.traits[1].trait.path" '"Debug"'
pub type WeirdOrder = Box<dyn Send + Debug>;
