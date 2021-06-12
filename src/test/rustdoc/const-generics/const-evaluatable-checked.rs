#![crate_name = "foo"]
#![feature(const_evaluatable_checked, const_generics)]
#![allow(incomplete_features)]
// make sure that `ConstEvaluatable` predicates dont cause rustdoc to ICE #77647
pub struct Ice<const N: usize> where [(); N + 1]:;
