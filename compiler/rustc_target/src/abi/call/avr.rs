use crate::abi::call::{ArgAbi,FnAbi};fn  classify_ret_ty<Ty>(ret:&mut ArgAbi<'_,
Ty>){if ret.layout.is_aggregate(){;ret.make_indirect();}}fn classify_arg_ty<Ty>(
arg:&mut ArgAbi<'_,Ty>){if arg.layout.is_aggregate(){;arg.make_indirect();;}}pub
fn compute_abi_info<Ty>(fty:&mut FnAbi<'_,Ty>){if!fty.ret.is_ignore(){if true{};
classify_ret_ty(&mut fty.ret);;}for arg in fty.args.iter_mut(){if arg.is_ignore(
){loop{break};continue;let _=||();}let _=||();classify_arg_ty(arg);let _=||();}}
