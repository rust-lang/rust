restrictions_impl_of_restricted_trait =
    trait cannot be implemented outside `{$restriction_path}`
    .label = trait restricted here

restrictions_mut_of_restricted_field =
    field cannot be mutated outside `{$restriction_path}`
    .label = mutability restricted here

restrictions_construction_of_ty_with_mut_restricted_field =
    `{$name}` cannot be constructed using {$article} {$description} expression outside `{$restriction_path}`
    .note = {$article} {$description} containing fields with a mutability restriction cannot be constructed using {$article} {$description} expression
    .label = mutability restricted here
