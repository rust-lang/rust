//@ set result = "$.index[?(@.name=='Result')].id"
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

//@ set my_error = "$.index[?(@.name=='MyError')].id"
pub struct MyError {}

//@ has   "$.index[?(@.name=='MyResult')].inner.type_alias"
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.where_predicates[*]" 0
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[*]" 2
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].name" \"T\"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].name" \"E\"
//@ has   "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type"
//@ has   "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type"
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type.bounds[*]" 0
//@ count "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.bounds[*]" 0
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[0].kind.type.default" null
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.generics.params[1].kind.type.default" 15
//@ has   "$.types[15].resolved_path"
//@ is    "$.types[15].resolved_path.id" $my_error
//@ is    "$.types[15].resolved_path.path" \"MyError\"
//@ is    "$.index[?(@.name=='MyResult')].inner.type_alias.type" 2
//@ has   "$.types[2].resolved_path"
//@ is    "$.types[2].resolved_path.id" $result
//@ is    "$.types[2].resolved_path.path" \"Result\"
//@ is    "$.types[2].resolved_path.args.angle_bracketed.constraints" []
//@ is    "$.types[2].resolved_path.args.angle_bracketed.args[0].type" 0
//@ is    "$.types[2].resolved_path.args.angle_bracketed.args[1].type" 1
//@ has   "$.types[0].generic"
//@ has   "$.types[1].generic"
//@ is    "$.types[0].generic" \"T\"
//@ is    "$.types[1].generic" \"E\"
pub type MyResult<T, E = MyError> = Result<T, E>;
