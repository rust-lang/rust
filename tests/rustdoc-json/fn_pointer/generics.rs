//@ arg with_higher_rank_trait_bounds .index[] | select(.name == "WithHigherRankTraitBounds").inner.type_alias.type?.function_pointer
//@ jq $with_higher_rank_trait_bounds.sig?.inputs[] | .[0] == "val" and .[1].borrowed_ref.lifetime? == "'c"
//@ jq $with_higher_rank_trait_bounds.sig?.output.primitive == "i32"
//@ jq $with_higher_rank_trait_bounds.generic_params[]? | .name == "'c" and .kind == {"lifetime": {"outlives": []}}
pub type WithHigherRankTraitBounds = for<'c> fn(val: &'c i32) -> i32;
