//! Module that define a common trait for things that represent a crate definition,
//! such as, a function, a trait, an enum, and any other definitions.

use crate::ty::Span;
use crate::{with, Crate, Symbol};

/// A unique identification number for each item accessible for the current compilation unit.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub(crate) usize);

/// A trait for retrieving information about a particular definition.
///
/// Implementors must provide the implementation of `def_id` which will be used to retrieve
/// information about a crate's definition.
pub trait CrateDef {
    /// Retrieve the unique identifier for the current definition.
    fn def_id(&self) -> DefId;

    /// Return the fully qualified name of the current definition.
    fn name(&self) -> Symbol {
        let def_id = self.def_id();
        with(|cx| cx.def_name(def_id, false))
    }

    /// Return a trimmed name of this definition.
    ///
    /// This can be used to print more user friendly diagnostic messages.
    ///
    /// If a symbol name can only be imported from one place for a type, and as
    /// long as it was not glob-imported anywhere in the current crate, we trim its
    /// path and print only the name.
    ///
    /// For example, this function may shorten `std::vec::Vec` to just `Vec`,
    /// as long as there is no other `Vec` importable anywhere.
    fn trimmed_name(&self) -> Symbol {
        let def_id = self.def_id();
        with(|cx| cx.def_name(def_id, true))
    }

    /// Return information about the crate where this definition is declared.
    ///
    /// This will return the crate number and its name.
    fn krate(&self) -> Crate {
        let def_id = self.def_id();
        with(|cx| cx.krate(def_id))
    }

    /// Return the span of this definition.
    fn span(&self) -> Span {
        let def_id = self.def_id();
        with(|cx| cx.span_of_an_item(def_id))
    }
}

macro_rules! crate_def {
    ( $(#[$attr:meta])*
      $vis:vis $name:ident $(;)?
    ) => {
        $(#[$attr])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        $vis struct $name(pub DefId);

        impl CrateDef for $name {
            fn def_id(&self) -> DefId {
                self.0
            }
        }
    };
}
