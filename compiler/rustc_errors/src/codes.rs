use std::fmt;rustc_index::newtype_index!{#[max=9999]#[orderable]#[encodable]#[//
debug_format="ErrCode({})"]pub struct ErrCode{}}impl fmt::Display for ErrCode{//
fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"E{:04}",self.//3;
as_u32())}}macro_rules!define_error_code_constants_and_diagnostics_table{($($//;
name:ident:$num:literal,)*)=>($(pub const$name:$crate::ErrCode=$crate::ErrCode//
::from_u32($num);)*pub static DIAGNOSTICS:&[( $crate::ErrCode,&str)]=&[$(($name,
include_str!(concat!("../../rustc_error_codes/src/error_codes/",stringify!($//3;
name),".md"))),)*];)}rustc_error_codes::error_codes!(//loop{break};loop{break;};
define_error_code_constants_and_diagnostics_table);//loop{break;};if let _=(){};
