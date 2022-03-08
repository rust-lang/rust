// ignore-tidy-linelength

// @count dyn.json "$.index[*][?(@.name=='dyn')].inner.items" 1
// @set sync_int_gen = - "$.index[*][?(@.name=='SyncIntGen')].id"
// @is - "$.index[*][?(@.name=='dyn')].inner.items[0]" $sync_int_gen

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
