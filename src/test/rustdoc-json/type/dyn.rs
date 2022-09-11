// ignore-tidy-linelength
use std::fmt::Debug;

// @count "$.index[*][?(@.name=='dyn')].inner.items[*]" 3
// @set sync_int_gen = "$.index[*][?(@.name=='SyncIntGen')].id"
// @set ref_fn       = "$.index[*][?(@.name=='RefFn')].id"
// @set weird_order  = "$.index[*][?(@.name=='WeirdOrder')].id"
// @ismany "$.index[*][?(@.name=='dyn')].inner.items[*]" $sync_int_gen $ref_fn $weird_order

// @is    "$.index[*][?(@.name=='SyncIntGen')].kind" \"typedef\"
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.generics" '{"params": [], "where_predicates": []}'
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.kind" \"resolved_path\"
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.name" \"Box\"
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.bindings" []
// @count "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args" 1
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.kind" \"dyn_trait\"
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.lifetime" \"\'static\"
// @count "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[*]" 3
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[0].generic_params" []
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[1].generic_params" []
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[2].generic_params" []
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[0].trait.name" '"Fn"'
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[1].trait.name" '"Send"'
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[2].trait.name" '"Sync"'
// @is    "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[0].trait.args" '{"parenthesized": {"inputs": [],"output": {"inner": "i32","kind": "primitive"}}}'
pub type SyncIntGen = Box<dyn Fn() -> i32 + Send + Sync + 'static>;

// @is "$.index[*][?(@.name=='RefFn')].kind" \"typedef\"
// @is "$.index[*][?(@.name=='RefFn')].inner.generics" '{"params": [{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"}],"where_predicates": []}'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.mutable" 'false'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.lifetime" "\"'a\""
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.kind" '"dyn_trait"'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.lifetime" null
// @count "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[*]" 1
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].trait.name" '"Fn"'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].trait.args.parenthesized.inputs[0].kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].trait.args.parenthesized.inputs[0].inner.lifetime" "\"'b\""
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].trait.args.parenthesized.output.kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.traits[0].trait.args.parenthesized.output.inner.lifetime" "\"'b\""
pub type RefFn<'a> = &'a dyn for<'b> Fn(&'b i32) -> &'b i32;

// @is    "$.index[*][?(@.name=='WeirdOrder')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[0].trait.name" '"Send"'
// @is    "$.index[*][?(@.name=='WeirdOrder')].inner.type.inner.args.angle_bracketed.args[0].type.inner.traits[1].trait.name" '"Debug"'
pub type WeirdOrder = Box<dyn Send + Debug>;
