#![crate_name = "foo"]
#![feature(const_evaluatable_checked, const_generics)]
#![allow(incomplete_features)]
// make sure that `ConstEvaluatable` predicates dont cause rustdoc to ICE #77647
// @has foo/struct.Ice.html '//pre[@class="rust struct"]' \
//      'pub struct Ice<const N: usize> where [(); N + 1]: ;'
pub struct Ice<const N: usize> where [(); N + 1]:;
