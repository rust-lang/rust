// ignore-tidy-linelength

// @count dyn.json "$.index[*][?(@.name=='dyn')].inner.items[*]" 2
// @set sync_int_gen = - "$.index[*][?(@.name=='SyncIntGen')].id"
// @set ref_fn = - "$.index[*][?(@.name=='RefFn')].id"
// @has - "$.index[*][?(@.name=='dyn')].inner.items[*]" $sync_int_gen
// @has - "$.index[*][?(@.name=='dyn')].inner.items[*]" $ref_fn

// @is    - "$.index[*][?(@.name=='SyncIntGen')].kind" \"typedef\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.generics" '{"params": [], "where_predicates": []}'
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.kind" \"resolved_path\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.name" \"Box\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.bindings" []
// @count - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args" 1
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.kind" \"resolved_path\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.kind" \"resolved_path\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.name" \"Fn\"
// @count - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.param_names[*]" 3
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.param_names[0].trait_bound.trait.inner.name" \"Send\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.param_names[1].trait_bound.trait.inner.name" \"Sync\"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.param_names[2]" "{\"outlives\": \"'static\"}"
// @is    - "$.index[*][?(@.name=='SyncIntGen')].inner.type.inner.args.angle_bracketed.args[0].type.inner.args" '{"parenthesized": {"inputs": [],"output": {"inner": "i32","kind": "primitive"}}}'
pub type SyncIntGen = Box<dyn Fn() -> i32 + Send + Sync + 'static>;

// @is - "$.index[*][?(@.name=='RefFn')].kind" \"typedef\"
// @is - "$.index[*][?(@.name=='RefFn')].inner.generics" '{"params": [{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"}],"where_predicates": []}'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.mutable" 'false'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.lifetime" "\"'a\""
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.kind" '"resolved_path"'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.name" '"Fn"'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.args.parenthesized.inputs[0].kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.args.parenthesized.inputs[0].inner.lifetime" "\"'b\""
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.args.parenthesized.output.kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='RefFn')].inner.type.inner.type.inner.args.parenthesized.output.inner.lifetime" "\"'b\""
pub type RefFn<'a> = &'a dyn for<'b> Fn(&'b i32) -> &'b i32;
