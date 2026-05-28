//@ set result = "$.index[?(@.name=='Result')].id"
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

//@ set my_error = "$.index[?(@.name=='MyError')].id"
pub struct MyError {}

//@ has    "$.index[?(@.name=='MyResult')].inner.type_alias"
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.where_predicates[*]" 0
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[*]" 2
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].name" \"T\"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].name" \"E\"
//@ has   "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type"
//@ has   "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type"
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type.bounds[*]" 0
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.bounds[*]" 0
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type.default" null
//@ has    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.default.resolved_path"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.default.resolved_path.id" $my_error
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.default.resolved_path.path" \"MyError\"
//@ has    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.id" $result
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.path" \"Result\"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.args.angle_bracketed.constraints" []
//@ has    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.generic"
//@ has    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[1].type.generic"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[0].type.generic" \"T\"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type.resolved_path.args.angle_bracketed.args[1].type.generic" \"E\"
pub type MyResult<T, E = MyError> = Result<T, E>;
