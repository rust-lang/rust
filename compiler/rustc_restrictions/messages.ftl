restrictions_impl_of_restricted_trait = implementation of restricted trait
    .label = trait restricted here

restrictions_mut_of_restricted_field = mutable use of restricted field
    .label = mutability restricted here

restrictions_construction_of_ty_with_mut_restricted_field = construction of {$ty} with mut restricted field
    .label = mutability restricted here
    .note = {$ty} expressions cannot be used when the {$ty} has a field with a mutability restriction
