#![allow(dead_code)]

// error-pattern:`main` function not found in crate

struct Tableau<'a, MP> {
    provider: &'a MP,
}

impl<'adapted_matrix_provider, 'original_data, MP>
    Tableau<'adapted_matrix_provider, AdaptedMatrixProvider<'original_data, MP>>
{
    fn provider(&self) -> &'adapted_matrix_provider AdaptedMatrixProvider</*'original_data,*/ MP> {
        self.provider
    }
}

struct AdaptedMatrixProvider<'a, T> {
    original_problem: &'a T,
}

impl<'a, T> AdaptedMatrixProvider<'a, T> {
    fn clone_with_extra_bound(&self) -> Self {
        AdaptedMatrixProvider { original_problem: self.original_problem }
    }
}

fn create_and_solve_subproblems<'data_provider, 'original_data, MP>(
    tableau: Tableau<'data_provider, AdaptedMatrixProvider<'original_data, MP>>,
) {
    let _: AdaptedMatrixProvider<'original_data, MP> = tableau.provider().clone_with_extra_bound();
    //~^ ERROR lifetime mismatch
}
