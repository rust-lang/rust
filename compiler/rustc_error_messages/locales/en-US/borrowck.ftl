borrowck_move_unsized =
    cannot move a value of type `{$ty}`
    .label = the size of `{$ty}` cannot be statically determined

borrowck_higher_ranked_lifetime_error =
    higher-ranked lifetime error

borrowck_could_not_prove =
    could not prove `{$predicate}`

borrowck_could_not_normalize =
    could not normalize `{$value}`

borrowck_higher_ranked_subtype_error =
    higher-ranked subtype error
  
generic_does_not_live_long_enough =
    `{$kind}` does not live long enough
    
borrowck_move_borrowed = 
    cannot move out of `{$desc}` beacause it is borrowed
    
borrowck_var_better_not_mut = 
    variable does not need to be mutable
    .suggestion = remove this `mut`

borrowck_const_not_used_in_type_alias = 
    const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias
