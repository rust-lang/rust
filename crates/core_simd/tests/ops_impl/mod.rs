#[macro_use]
#[path = "../helpers/mod.rs"]
mod helpers;

#[macro_use]
mod int_macros;

mod r#i8;
mod r#i16;
mod r#i32;
mod r#i64;
mod r#i128;
mod r#isize;

#[macro_use]
mod uint_macros;

mod r#u8;
mod r#u16;
mod r#u32;
mod r#u64;
mod r#u128;
mod r#usize;

#[macro_use]
mod mask_macros;

mod mask8;
mod mask16;
mod mask32;
mod mask64;
mod mask128;
mod masksize;
