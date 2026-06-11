use crate::common::constraint::Constraint;

use super::argument::ArgumentList;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic<T: IntrinsicTypeDefinition> {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList<T>,

    /// The return type of this intrinsic.
    pub results: T,

    /// Any architecture-specific tags.
    pub arch_tags: Vec<String>,

    /// Specific extension that the intrinsic is from
    pub extension: String,
}

/// Invokes `f` for each combination of the values in the constraint ranges.
///
/// For example, given `constraints=[Equal(0), Range(1..2), Set([3, 4])]` and `imm_values=[]`, this
/// produces the four calls to `f`: `f([0, 1, 3])`, `f([0, 1, 4])`, `f([0, 2, 3])`, `f([0, 2, 4])`.
fn recurse_specializations<'a, E>(
    constraints: &mut (impl Iterator<Item = &'a Constraint> + Clone),
    imm_values: &mut Vec<i64>,
    f: &mut impl FnMut(&[i64]) -> Result<(), E>,
) -> Result<(), E> {
    if let Some(current) = constraints.next() {
        for i in current.iter() {
            imm_values.push(i);
            recurse_specializations(&mut constraints.clone(), imm_values, f)?;
            imm_values.pop();
        }
        Ok(())
    } else {
        f(&imm_values)
    }
}

impl<T: IntrinsicTypeDefinition> Intrinsic<T> {
    /// Invokes `f` for "specialisation" of the intrinsic - a specific instantiation of the
    /// constant generics of the intrinsic. `f` takes a slice where the `i`th element corresponds
    /// to the value of the `i`th const generic argument of the intrinsic.
    ///
    /// For an intrinsic with three arguments with constraints `Equal(0)`, `Range(1..2)`,
    /// `Set([3, 4])` respectively, this would produce four calls to `f`: `f(0, 1, 3)`,
    /// `f(0, 1, 4)`, `f(0, 2, 3)`, `f(0, 2, 4)`.
    pub fn iter_specializations<E>(
        &self,
        mut f: impl FnMut(&[i64]) -> Result<(), E>,
    ) -> Result<(), E> {
        recurse_specializations(
            &mut self
                .arguments
                .iter()
                .filter_map(|arg| arg.constraint.as_ref()),
            &mut Vec::new(),
            &mut f,
        )
    }
}
