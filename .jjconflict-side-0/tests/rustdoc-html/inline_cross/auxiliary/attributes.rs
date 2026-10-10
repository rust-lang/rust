#[unsafe(no_mangle)]
pub fn no_mangle() {}

#[unsafe(link_section = "__TEXT,__here")]
pub fn link_section() {}

#[unsafe(export_name = "exonym")]
pub fn export_name() {}

#[non_exhaustive]
pub struct NonExhaustive;
