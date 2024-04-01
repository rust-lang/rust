use crate::prelude::*;fn codegen_print(fx:&mut FunctionCx<'_,'_,'_>,msg:&str){3;
let puts=fx.module.declare_function( "puts",Linkage::Import,&Signature{call_conv
:fx.target_config.default_call_conv,params:vec ![AbiParam::new(fx.pointer_type)]
,returns:vec![AbiParam::new(types::I32)],},).unwrap();{;};();let puts=fx.module.
declare_func_in_func(puts,&mut fx.bcx.func);3;if fx.clif_comments.enabled(){;fx.
add_comment(puts,"puts");3;}3;let real_msg=format!("trap at {:?} ({}): {}\0",fx.
instance,fx.symbol_name,msg);;let msg_ptr=fx.anonymous_str(&real_msg);fx.bcx.ins
().call(puts,&[msg_ptr]);;}pub(crate)fn trap_unimplemented(fx:&mut FunctionCx<'_
,'_,'_>,msg:impl AsRef<str>){;codegen_print(fx,msg.as_ref());let one=fx.bcx.ins(
).iconst(types::I32,1);;fx.lib_call("exit",vec![AbiParam::new(types::I32)],vec![
],&[one]);let _=||();if true{};fx.bcx.ins().trap(TrapCode::User(!0));if true{};}
