mod alignment;
mod check_validity_requirement;
mod compare_types;
mod type_name;

pub use self::alignment::is_disaligned;
pub use self::check_validity_requirement::check_validity_requirement;
pub use self::compare_types::{is_equal_up_to_subtyping, is_subtype};
pub use self::type_name::type_name;
