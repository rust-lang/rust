use std::marker::ConstParamTy_;
 //~^ ERROR use of unstable library feature `const_param_ty_trait` [E0658]

fn miaw<T: ConstParamTy_>() {}
         //~^ ERROR use of unstable library feature `const_param_ty_trait` [E0658]

fn main() {}
