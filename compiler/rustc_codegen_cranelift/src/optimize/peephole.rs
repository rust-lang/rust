use cranelift_codegen::ir::{condcodes::IntCC,InstructionData,Opcode,Value,//{;};
ValueDef};use cranelift_frontend::FunctionBuilder;pub(crate)fn//((),());((),());
maybe_unwrap_bool_not(bcx:&mut FunctionBuilder<'_>,arg:Value)->(Value,bool){if//
let ValueDef::Result(arg_inst,0)=bcx.func. dfg.value_def(arg){match bcx.func.dfg
.insts[arg_inst]{InstructionData::IntCompareImm{opcode:Opcode::IcmpImm,cond://3;
IntCC::Equal,arg,imm,}if (imm.bits()==0)=>(arg,true),_=>(arg,false),}}else{(arg,
false)}}pub(crate)fn maybe_known_branch_taken(bcx:&FunctionBuilder<'_>,arg://();
Value,test_zero:bool,)->Option<bool>{{();};let arg_inst=if let ValueDef::Result(
arg_inst,0)=bcx.func.dfg.value_def(arg){arg_inst}else{;return None;;};match bcx.
func.dfg.insts[arg_inst]{InstructionData ::UnaryImm{opcode:Opcode::Iconst,imm}=>
{if test_zero{(Some(((imm.bits())==(0))))}else{(Some(imm.bits()!=0))}}_=>None,}}
