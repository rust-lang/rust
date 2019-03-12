use super::*;
use std::fmt::{self, Display};

impl Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = &self.visibility {
            write!(f, "{} ", t)?;
        }

        if let Some(name) = &self.name {
            write!(f, "fn {}", name)?;
        }

        if !self.generic_parameters.is_empty() {
            write!(f, "<")?;
            write_joined(f, &self.generic_parameters, ", ")?;
            write!(f, ">")?;
        }

        write!(f, "(")?;
        write_joined(f, &self.parameters, ", ")?;
        write!(f, ")")?;

        if let Some(t) = &self.ret_type {
            write!(f, " -> {}", t)?;
        }

        if !self.where_predicates.is_empty() {
            write!(f, "\nwhere ")?;
            write_joined(f, &self.where_predicates, ",\n      ")?;
        }

        Ok(())
    }
}

fn write_joined<T: Display>(
    f: &mut fmt::Formatter,
    items: impl IntoIterator<Item = T>,
    sep: &str,
) -> fmt::Result {
    let mut first = true;
    for e in items {
        if !first {
            write!(f, "{}", sep)?;
        }
        first = false;
        write!(f, "{}", e)?;
    }
    Ok(())
}
