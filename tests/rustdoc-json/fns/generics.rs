//@ arg wham_id .index[] | select(.name == "Wham").id
pub trait Wham {}

//@ arg one_generic_param_fn .index[] | select(.name == "one_generic_param_fn").inner.function
//@ jq $one_generic_param_fn.generics? | .where_predicates == [] and .params[].name == "T"
//@ jq $one_generic_param_fn.generics?.params[].kind.type | .is_synthetic? == false and .bounds?[].trait_bound.trait?.id == $wham_id
//@ jq $one_generic_param_fn.sig?.inputs == [["w", {"generic": "T"}]]
pub fn one_generic_param_fn<T: Wham>(w: T) {}

//@ arg one_synthetic_generic_param_fn .index[] | select(.name == "one_synthetic_generic_param_fn").inner.function
//@ jq $one_synthetic_generic_param_fn.generics? | .where_predicates == [] and .params[].name == "impl Wham"
//@ jq $one_synthetic_generic_param_fn.generics?.params[].kind.type | .is_synthetic? == true and .bounds?[].trait_bound.trait?.id == $wham_id
//@ jq $one_synthetic_generic_param_fn.sig?.inputs[] | .[0] == "w" and .[1].impl_trait[]?.trait_bound.trait?.id == $wham_id
pub fn one_synthetic_generic_param_fn(w: impl Wham) {}
