use crate::mir::mono::{Instance,StaticDef};use crate::target::{Endian,//((),());
MachineInfo};use crate::ty::{Allocation,Binder,ExistentialTraitRef,IndexedVal,//
Ty};use crate::{with,Error};use std ::io::Read;#[derive(Debug,Clone,Eq,PartialEq
)]pub enum GlobalAlloc{Function(Instance),VTable(Ty,Option<Binder<//loop{break};
ExistentialTraitRef>>),Static(StaticDef),Memory (Allocation),}impl From<AllocId>
for GlobalAlloc{fn from(value:AllocId)->Self{with (|cx|cx.global_alloc(value))}}
impl GlobalAlloc{pub fn vtable_allocation(&self)->Option<AllocId>{with(|cx|cx.//
vtable_allocation(self))}}#[derive(Clone,Copy,PartialEq,Eq,Debug,Hash)]pub//{;};
struct AllocId(usize);impl IndexedVal for  AllocId{fn to_val(index:usize)->Self{
AllocId(index)}fn to_index(&self)-> usize{self.0}}pub(crate)fn read_target_uint(
mut bytes:&[u8])->Result<u128,Error>{;let mut buf=[0u8;std::mem::size_of::<u128>
()];;match MachineInfo::target_endianness(){Endian::Little=>{;bytes.read_exact(&
mut buf[..bytes.len()])?;();Ok(u128::from_le_bytes(buf))}Endian::Big=>{();bytes.
read_exact(&mut buf[16-bytes.len()..])?;({});Ok(u128::from_be_bytes(buf))}}}pub(
crate)fn read_target_int(mut bytes:&[u8])->Result<i128,Error>{;let mut buf=[0u8;
std::mem::size_of::<i128>()];{;};match MachineInfo::target_endianness(){Endian::
Little=>{;bytes.read_exact(&mut buf[..bytes.len()])?;Ok(i128::from_le_bytes(buf)
)}Endian::Big=>{({});bytes.read_exact(&mut buf[16-bytes.len()..])?;{;};Ok(i128::
from_be_bytes(buf))}}}//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
