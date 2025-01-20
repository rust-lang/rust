//! You can find more docs on what groups are on [`AttributeParser`] itself.
//! However, for many types of attributes, implementing [`AttributeParser`] is not necessary.
//! It allows for a lot of flexibility you might not want.
//!
//! Specifically, you might not care about managing the state of your [`AttributeParser`]
//! state machine yourself. In this case you can choose to implement:
//!
//! - [`SingleAttributeParser`]: makes it easy to implement an attribute which should error if it
//! appears more than once in a list of attributes
//! - [`CombineAttributeParser`]: makes it easy to implement an attribute which should combine the
//! contents of attributes, if an attribute appear multiple times in a list
//!
//! Attributes should be added to [`ATTRIBUTE_GROUP_MAPPING`](crate::context::ATTRIBUTE_GROUP_MAPPING) to be parsed.

pub(crate) mod allow_unstable;
pub(crate) mod cfg;
pub(crate) mod confusables;
pub(crate) mod deprecation;
pub(crate) mod repr;
pub(crate) mod stability;
pub(crate) mod transparency;
pub(crate) mod util;

pub use allow_unstable::*;
pub use cfg::*;
pub use confusables::*;
pub use deprecation::*;
pub use repr::*;
pub use stability::*;
pub use transparency::*;
