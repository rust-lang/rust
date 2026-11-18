#![feature(register_tool)]
#![feature(type_info)]

#![register_tool(aaa)]
#![register_tool(bbb(reflect))]

#[aaa::aa]
#[bbb::bb]
pub struct A;
