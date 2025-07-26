#[unsafe(no_mangle)]
pub fn f0() {}

#[unsafe(link_section = ".here")]
pub fn f1() {}

#[unsafe(export_name = "f2export")]
pub fn f2() {}

#[repr(u8)]
pub enum T0 { V1 }

#[non_exhaustive]
pub enum T1 {}
