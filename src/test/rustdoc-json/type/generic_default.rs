// ignore-tidy-linelength

// @set result = generic_default.json "$.index[*][?(@.name=='Result')].id"
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

// @set my_error = - "$.index[*][?(@.name=='MyError')].id"
pub struct MyError {}

// @is    - "$.index[*][?(@.name=='MyResult')].kind" \"typedef\"
// @count - "$.index[*][?(@.name=='MyResult')].inner.generics.where_predicates[*]" 0
// @count - "$.index[*][?(@.name=='MyResult')].inner.generics.params[*]" 2
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[0].name" \"T\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].name" \"E\"
// @has   - "$.index[*][?(@.name=='MyResult')].inner.generics.params[0].kind.type"
// @has   - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].kind.type"
// @count - "$.index[*][?(@.name=='MyResult')].inner.generics.params[0].kind.type.bounds[*]" 0
// @count - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].kind.type.bounds[*]" 0
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[0].kind.type.default" null
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].kind.type.default.kind" \"resolved_path\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].kind.type.default.inner.id" $my_error
// @is    - "$.index[*][?(@.name=='MyResult')].inner.generics.params[1].kind.type.default.inner.name" \"MyError\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.kind" \"resolved_path\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.id" $result
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.name" \"Result\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.args.angle_bracketed.bindings" []
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.args.angle_bracketed.args[0].type.kind" \"generic\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.args.angle_bracketed.args[1].type.kind" \"generic\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.args.angle_bracketed.args[0].type.inner" \"T\"
// @is    - "$.index[*][?(@.name=='MyResult')].inner.type.inner.args.angle_bracketed.args[1].type.inner" \"E\"
pub type MyResult<T, E = MyError> = Result<T, E>;
