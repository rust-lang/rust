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
}

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
