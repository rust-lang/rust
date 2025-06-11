use std::fmt::Debug;

//@ count "$.index[?(@.name=='dyn')].inner.module.items[*]" 3
//@ set sync_int_gen = "$.index[?(@.name=='SyncIntGen')].id"
//@ set ref_fn       = "$.index[?(@.name=='RefFn')].id"
//@ set weird_order  = "$.index[?(@.name=='WeirdOrder')].id"
//@ ismany "$.index[?(@.name=='dyn')].inner.module.items[*]" $sync_int_gen $ref_fn $weird_order

//@ has   "$.index[?(@.name=='SyncIntGen')].inner.type_alias"
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.generics" '{"params": [], "where_predicates": []}'
//@ is    "$.index[?(@.name=='SyncIntGen')].inner.type_alias.type" 2
//@ has   "$.types[2].resolved_path"
//@ is    "$.types[2].resolved_path.path" \"Box\"
//@ is    "$.types[2].resolved_path.args.angle_bracketed.constraints" []
//@ count "$.types[2].resolved_path.args.angle_bracketed.args" 1
//@ is    "$.types[2].resolved_path.args.angle_bracketed.args[0].type" 1
//@ has   "$.types[1].dyn_trait"
//@ is    "$.types[1].dyn_trait.lifetime" \"\'static\"
//@ count "$.types[1].dyn_trait.traits[*]" 3
//@ is    "$.types[1].dyn_trait.traits[0].generic_params" []
//@ is    "$.types[1].dyn_trait.traits[1].generic_params" []
//@ is    "$.types[1].dyn_trait.traits[2].generic_params" []
//@ is    "$.types[1].dyn_trait.traits[0].trait.path" '"Fn"'
//@ is    "$.types[1].dyn_trait.traits[1].trait.path" '"Send"'
//@ is    "$.types[1].dyn_trait.traits[2].trait.path" '"Sync"'
//@ is    "$.types[0].primitive" '"i32"'
//@ is    "$.types[1].dyn_trait.traits[0].trait.args" '{"parenthesized": {"inputs": [],"output": 0}}'
pub type SyncIntGen = Box<dyn Fn() -> i32 + Send + Sync + 'static>;

//@ has "$.index[?(@.name=='RefFn')].inner.type_alias"
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.generics" '{"params": [{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"}],"where_predicates": []}'
//@ is "$.index[?(@.name=='RefFn')].inner.type_alias.type" 5
//@ has "$.types[5].borrowed_ref"
//@ is "$.types[5].borrowed_ref.is_mutable" 'false'
//@ is "$.types[5].borrowed_ref.lifetime" "\"'a\""
//@ is "$.types[5].borrowed_ref.type" 4
//@ has "$.types[4].dyn_trait"
//@ is "$.types[4].dyn_trait.lifetime" null
//@ count "$.types[4].dyn_trait.traits[*]" 1
//@ is "$.types[4].dyn_trait.traits[0].generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
//@ is "$.types[4].dyn_trait.traits[0].trait.path" '"Fn"'
//@ is "$.types[4].dyn_trait.traits[0].trait.args.parenthesized.inputs[0]" 3
//@ has "$.types[3].borrowed_ref"
//@ is "$.types[3].borrowed_ref.lifetime" "\"'b\""
//@ is "$.types[4].dyn_trait.traits[0].trait.args.parenthesized.output" 3
pub type RefFn<'a> = &'a dyn for<'b> Fn(&'b i32) -> &'b i32;

//@ is "$.index[?(@.name=='WeirdOrder')].inner.type_alias.type" 7
//@ is "$.types[7].resolved_path.args.angle_bracketed.args[0].type" 6
//@ is "$.types[6].dyn_trait.traits[0].trait.path" '"Send"'
//@ is "$.types[6].dyn_trait.traits[1].trait.path" '"Debug"'
pub type WeirdOrder = Box<dyn Send + Debug>;
