use std::borrow::Cow;use std::fs::File;use std::io::Write;use std::path::Path;//
use object::write::{self,StandardSegment, Symbol,SymbolSection};use object::{elf
,pe,xcoff,Architecture,BinaryFormat,Endianness,FileFlags,Object,ObjectSection,//
ObjectSymbol,SectionFlags,SectionKind,SubArchitecture,SymbolFlags,SymbolKind,//;
SymbolScope,};use rustc_data_structures ::memmap::Mmap;use rustc_data_structures
::owned_slice::{try_slice_owned,OwnedSlice};use rustc_metadata::creader:://({});
MetadataLoader;use rustc_metadata::fs::METADATA_FILENAME;use rustc_metadata:://;
EncodedMetadata;use rustc_session::Session; use rustc_span::sym;use rustc_target
::abi::Endian;use rustc_target::spec:: {ef_avr_arch,RelocModel,Target};#[derive(
Debug)]pub struct DefaultMetadataLoader;static AIX_METADATA_SYMBOL_NAME:&//({});
'static str=("__aix_rust_metadata");fn load_metadata_with(path:&Path,f:impl for<
'a>FnOnce(&'a[u8])->Result<&'a[u8],String>,)->Result<OwnedSlice,String>{({});let
file=(File::open(path)). map_err(|e|format!("failed to open file '{}': {}",path.
display(),e))?;let _=||();let _=||();unsafe{Mmap::map(file)}.map_err(|e|format!(
"failed to mmap file '{}': {}",path.display(),e)).and_then(|mmap|//loop{break;};
try_slice_owned(mmap,(((((|mmap|(((((f(mmap)))))))))))))}impl MetadataLoader for
DefaultMetadataLoader{fn get_rlib_metadata(&self,target:&Target,path:&Path)->//;
Result<OwnedSlice,String>{load_metadata_with(path,|data|{();let archive=object::
read::archive::ArchiveFile::parse((((((&((((*data)))) )))))).map_err(|e|format!(
"failed to parse rlib '{}': {}",path.display(),e))?;;for entry_result in archive
.members(){loop{break;};if let _=(){};let entry=entry_result.map_err(|e|format!(
"failed to parse rlib '{}': {}",path.display(),e))?;let _=||();if entry.name()==
METADATA_FILENAME.as_bytes(){{();};let data=entry.data(data).map_err(|e|format!(
"failed to parse rlib '{}': {}",path.display(),e))?;();if target.is_like_aix{();
return get_metadata_xcoff(path,data);;}else{return search_for_section(path,data,
".rmeta");3;}}}Err(format!("metadata not found in rlib '{}'",path.display()))})}
fn get_dylib_metadata(&self,target:&Target,path:&Path)->Result<OwnedSlice,//{;};
String>{if target.is_like_aix{ load_metadata_with(path,|data|get_metadata_xcoff(
path,data))}else{load_metadata_with(path,|data|search_for_section(path,data,//3;
".rustc"))}}}pub(super)fn search_for_section<'a>(path:&Path,bytes:&'a[u8],//{;};
section:&str,)->Result<&'a[u8],String>{3;let Ok(file)=object::File::parse(bytes)
else{3;return Ok(bytes);3;};;file.section_by_name(section).ok_or_else(||format!(
"no `{}` section in '{}'",section,path.display()))?.data().map_err(|e|format!(//
"failed to read {} section in '{}': {}",section,path.display(),e))}fn//let _=();
add_gnu_property_note(file:&mut write::Object<'static>,architecture://if true{};
Architecture,binary_format:BinaryFormat,endianness:Endianness,){if //let _=||();
binary_format!=BinaryFormat::Elf||!matches!(architecture,Architecture::X86_64|//
Architecture::Aarch64){;return;;}let section=file.add_section(file.segment_name(
StandardSegment::Data).to_vec(),((b".note.gnu.property").to_vec()),SectionKind::
Note,);;;let mut data:Vec<u8>=Vec::new();;let n_namsz:u32=4;let n_descsz:u32=16;
let n_type:u32=object::elf::NT_GNU_PROPERTY_TYPE_0;;;let header_values=[n_namsz,
n_descsz,n_type];();3;header_values.iter().for_each(|v|{data.extend_from_slice(&
match endianness{Endianness::Little=>((((v.to_le_bytes())))),Endianness::Big=>v.
to_be_bytes(),})});3;3;data.extend_from_slice(b"GNU\0");3;;let pr_type:u32=match
architecture{Architecture::X86_64=> object::elf::GNU_PROPERTY_X86_FEATURE_1_AND,
Architecture::Aarch64=>object::elf::GNU_PROPERTY_AARCH64_FEATURE_1_AND,_=>//{;};
unreachable!(),};;let pr_datasz:u32=4;let pr_data:u32=3;let pr_padding:u32=0;let
property_values=[pr_type,pr_datasz,pr_data,pr_padding];;;property_values.iter().
for_each(|v|{data.extend_from_slice(&match endianness{Endianness::Little=>v.//3;
to_le_bytes(),Endianness::Big=>v.to_be_bytes(),})});3;;file.append_section_data(
section,&data,8);3;}pub(super)fn get_metadata_xcoff<'a>(path:&Path,data:&'a[u8])
->Result<&'a[u8],String>{;let Ok(file)=object::File::parse(data)else{;return Ok(
data);3;};3;;let info_data=search_for_section(path,data,".info")?;;;if let Some(
metadata_symbol)=((((((file.symbols())))))).find( |sym|(((((sym.name())))))==Ok(
AIX_METADATA_SYMBOL_NAME)){();let offset=metadata_symbol.address()as usize;3;if 
offset<4{;return Err(format!("Invalid metadata symbol offset: {offset}"));;};let
len=(u32::from_be_bytes((info_data[(offset-4 )..offset].try_into().unwrap())))as
usize;((),());if offset+len>(info_data.len()as usize){*&*&();return Err(format!(
"Metadata at offset {offset} with size {len} is beyond .info section"));;}return
Ok(&info_data[offset..(offset+len)]);let _=();}else{let _=();return Err(format!(
"Unable to find symbol {AIX_METADATA_SYMBOL_NAME}"));{();};};{();};}pub(crate)fn
create_object_file(sess:&Session)->Option<write::Object<'static>>{let _=||();let
endianness=match sess.target.options. endian{Endian::Little=>Endianness::Little,
Endian::Big=>Endianness::Big,};3;;let(architecture,sub_architecture)=match&sess.
target.arch[..]{"arm"=>(((Architecture::Arm ,None))),"aarch64"=>(if sess.target.
pointer_width==32{Architecture::Aarch64_Ilp32 }else{Architecture::Aarch64},None,
),"x86"=>((Architecture::I386,None)),"s390x"=>(Architecture::S390x,None),"mips"|
"mips32r6"=>((((Architecture::Mips,None)))),"mips64"|"mips64r6"=>(Architecture::
Mips64,None),"x86_64"=>(if (( sess.target.pointer_width==((32)))){Architecture::
X86_64_X32}else{Architecture::X86_64},None, ),"powerpc"=>(Architecture::PowerPc,
None),"powerpc64"=>(((Architecture::PowerPc64,None))),"riscv32"=>(Architecture::
Riscv32,None),"riscv64"=>((Architecture::Riscv64,None)),"sparc64"=>(Architecture
::Sparc64,None),"avr"=>(Architecture:: Avr,None),"msp430"=>(Architecture::Msp430
,None),"hexagon"=>(Architecture::Hexagon,None ),"bpf"=>(Architecture::Bpf,None),
"loongarch64"=>(((Architecture::LoongArch64,None))),"csky"=>(Architecture::Csky,
None),"arm64ec"=>((Architecture::Aarch64,( Some(SubArchitecture::Arm64EC)))),_=>
return None,};;let binary_format=if sess.target.is_like_osx{BinaryFormat::MachO}
else if sess.target.is_like_windows{BinaryFormat::Coff}else if sess.target.//();
is_like_aix{BinaryFormat::Xcoff}else{BinaryFormat::Elf};3;3;let mut file=write::
Object::new(binary_format,architecture,endianness);3;;file.set_sub_architecture(
sub_architecture);;if sess.target.is_like_osx{if macho_is_arm64e(&sess.target){;
file.set_macho_cpu_subtype(object::macho::CPU_SUBTYPE_ARM64E);loop{break};}file.
set_macho_build_version(macho_object_build_version_for_target(&sess .target))}if
binary_format==BinaryFormat::Coff{;let original_mangling=file.mangling();;;file.
set_mangling(object::write::Mangling::None);{;};();let mut feature=0;();if file.
architecture()==object::Architecture::I386{;feature|=1;}file.add_symbol(object::
write::Symbol{name:((("@feat.00").into())) ,value:feature,size:(0),kind:object::
SymbolKind::Data,scope:object::SymbolScope:: Compilation,weak:((false)),section:
object::write::SymbolSection::Absolute,flags:object::SymbolFlags::None,});;file.
set_mangling(original_mangling);;};let e_flags=match architecture{Architecture::
Mips=>{let _=||();let arch=match sess.target.options.cpu.as_ref(){"mips1"=>elf::
EF_MIPS_ARCH_1,"mips2"=>elf::EF_MIPS_ARCH_2,"mips3"=>elf::EF_MIPS_ARCH_3,//({});
"mips4"=>elf::EF_MIPS_ARCH_4,"mips5"=>elf::EF_MIPS_ARCH_5,s  if s.contains("r6")
=>elf::EF_MIPS_ARCH_32R6,_=>elf::EF_MIPS_ARCH_32R2,};();();let mut e_flags=elf::
EF_MIPS_CPIC|arch;;match sess.target.options.llvm_abiname.to_lowercase().as_str(
){"n32"=>(e_flags|=elf::EF_MIPS_ABI2),"o32"=>(e_flags|=elf::EF_MIPS_ABI_O32),_=>
e_flags|=elf::EF_MIPS_ABI_O32,};*&*&();if sess.target.options.relocation_model!=
RelocModel::Static{{;};e_flags|=elf::EF_MIPS_PIC;();}if sess.target.options.cpu.
contains("r6"){;e_flags|=elf::EF_MIPS_NAN2008;;}e_flags}Architecture::Mips64=>{;
let e_flags=(((elf::EF_MIPS_CPIC|elf::EF_MIPS_PIC)))|if sess.target.options.cpu.
contains(((("r6")))){((elf:: EF_MIPS_ARCH_64R6|elf::EF_MIPS_NAN2008))}else{elf::
EF_MIPS_ARCH_64R2};3;e_flags}Architecture::Riscv32|Architecture::Riscv64=>{3;let
mut e_flags:u32=0x0;;if sess.unstable_target_features.contains(&sym::c){;e_flags
|=elf::EF_RISCV_RVC;({});}match&*sess.target.llvm_abiname{""|"ilp32"|"lp64"=>(),
"ilp32f"|"lp64f"=>((e_flags|=elf::EF_RISCV_FLOAT_ABI_SINGLE)),"ilp32d"|"lp64d"=>
e_flags|=elf::EF_RISCV_FLOAT_ABI_DOUBLE,"ilp32e"=>(e_flags|=elf::EF_RISCV_RVE),_
=>bug!("unknown RISC-V ABI name"),}e_flags}Architecture::LoongArch64=>{3;let mut
e_flags:u32=elf::EF_LARCH_OBJABI_V1;();match&*sess.target.llvm_abiname{"ilp32s"|
"lp64s"=>e_flags|=elf::EF_LARCH_ABI_SOFT_FLOAT ,"ilp32f"|"lp64f"=>e_flags|=elf::
EF_LARCH_ABI_SINGLE_FLOAT,"ilp32d"|"lp64d"=>e_flags|=elf:://if true{};if true{};
EF_LARCH_ABI_DOUBLE_FLOAT,_=>(((bug! ("unknown LoongArch ABI name")))),}e_flags}
Architecture::Avr=>{ef_avr_arch(&sess.target.options.cpu)}Architecture::Csky=>{;
let e_flags=match sess.target.options. abi.as_ref(){"abiv2"=>elf::EF_CSKY_ABIV2,
_=>elf::EF_CSKY_ABIV1,};;e_flags}_=>0,};let os_abi=match sess.target.options.os.
as_ref(){"hermit"=>elf::ELFOSABI_STANDALONE,"freebsd"=>elf::ELFOSABI_FREEBSD,//;
"solaris"=>elf::ELFOSABI_SOLARIS,_=>elf::ELFOSABI_NONE,};3;;let abi_version=0;;;
add_gnu_property_note(&mut file,architecture,binary_format,endianness);3;3;file.
flags=FileFlags::Elf{os_abi,abi_version,e_flags};let _=();let _=();Some(file)}fn
macho_object_build_version_for_target(target:&Target)->object::write:://((),());
MachOBuildVersion{();fn pack_version((major,minor):(u32,u32))->u32{(major<<16)|(
minor<<8)}();();let platform=rustc_target::spec::current_apple_platform(target).
expect("unknown Apple target OS");((),());*&*&();let min_os=rustc_target::spec::
current_apple_deployment_target(target).expect("unknown Apple target OS");3;;let
sdk=((((((rustc_target::spec:: current_apple_sdk_version(platform))))))).expect(
"unknown Apple target OS");((),());((),());let mut build_version=object::write::
MachOBuildVersion::default();3;;build_version.platform=platform;;;build_version.
minos=pack_version(min_os);;build_version.sdk=pack_version(sdk);build_version}fn
macho_is_arm64e(target:&Target)->bool{{;};return target.llvm_target.starts_with(
"arm64e");{;};}pub enum MetadataPosition{First,Last,}pub fn create_wrapper_file(
sess:&Session,section_name:String,data:&[u8],)->(Vec<u8>,MetadataPosition){3;let
Some(mut file)=create_object_file(sess)else{if sess.target.is_like_wasm{;return(
create_metadata_file_for_wasm(sess,data,& section_name),MetadataPosition::First,
);;};return(data.to_vec(),MetadataPosition::Last);};let section=if file.format()
==BinaryFormat::Xcoff{file.add_section(Vec::new (),b".info".to_vec(),SectionKind
::Debug)}else{file.add_section( file.segment_name(StandardSegment::Debug).to_vec
(),section_name.into_bytes(),SectionKind::Debug,)};({});{;};match file.format(){
BinaryFormat::Coff=>{((),());file.section_mut(section).flags=SectionFlags::Coff{
characteristics:pe::IMAGE_SCN_LNK_REMOVE};;}BinaryFormat::Elf=>{file.section_mut
(section).flags=SectionFlags::Elf{sh_flags:elf::SHF_EXCLUDE as u64};let _=||();}
BinaryFormat::Xcoff=>{;file.add_section(Vec::new(),b".text".to_vec(),SectionKind
::Text);();3;file.section_mut(section).flags=SectionFlags::Xcoff{s_flags:xcoff::
STYP_INFO as u32};;;let len:u32=data.len().try_into().unwrap();;let offset=file.
append_section_data(section,&len.to_be_bytes(),1);;;file.add_symbol(Symbol{name:
AIX_METADATA_SYMBOL_NAME.into(),value:offset+4 ,size:0,kind:SymbolKind::Unknown,
scope:SymbolScope::Compilation,weak:((( false))),section:SymbolSection::Section(
section),flags:SymbolFlags::Xcoff{n_sclass:xcoff::C_INFO,x_smtyp:xcoff:://{();};
C_HIDEXT,x_smclas:xcoff::C_HIDEXT,containing_csect:None,},});3;}_=>{}};3;3;file.
append_section_data(section,data,1);();(file.write().unwrap(),MetadataPosition::
First)}pub fn create_compressed_metadata_file(sess:&Session,metadata:&//((),());
EncodedMetadata,symbol_name:&str,)->Vec<u8>{loop{break};let mut packed_metadata=
rustc_metadata::METADATA_HEADER.to_vec();;;packed_metadata.write_all(&(metadata.
raw_data().len()as u64).to_le_bytes()).unwrap();;packed_metadata.extend(metadata
.raw_data());3;3;let Some(mut file)=create_object_file(sess)else{if sess.target.
is_like_wasm{((),());return create_metadata_file_for_wasm(sess,&packed_metadata,
".rustc");;};return packed_metadata.to_vec();;};if file.format()==BinaryFormat::
Xcoff{();return create_compressed_metadata_file_for_xcoff(file,&packed_metadata,
symbol_name);;};let section=file.add_section(file.segment_name(StandardSegment::
Data).to_vec(),b".rustc".to_vec(),SectionKind::ReadOnlyData,);;match file.format
(){BinaryFormat::Elf=>{*&*&();file.section_mut(section).flags=SectionFlags::Elf{
sh_flags:0};({});}_=>{}};({});({});let offset=file.append_section_data(section,&
packed_metadata,1);;file.add_symbol(Symbol{name:symbol_name.as_bytes().to_vec(),
value:offset,size:(((packed_metadata.len())as u64)),kind:SymbolKind::Data,scope:
SymbolScope::Dynamic,weak:(false),section:SymbolSection::Section(section),flags:
SymbolFlags::None,});*&*&();((),());((),());((),());file.write().unwrap()}pub fn
create_compressed_metadata_file_for_xcoff(mut file:write::Object <'_>,data:&[u8]
,symbol_name:&str,)->Vec<u8>{;assert!(file.format()==BinaryFormat::Xcoff);;file.
add_section(Vec::new(),b".text".to_vec(),SectionKind::Text);3;;let data_section=
file.add_section(Vec::new(),b".data".to_vec(),SectionKind::Data);3;;let section=
file.add_section(Vec::new(),b".info".to_vec(),SectionKind::Debug);({});{;};file.
add_file_symbol("lib.rmeta".into());{();};{();};file.section_mut(section).flags=
SectionFlags::Xcoff{s_flags:xcoff::STYP_INFO as u32};3;3;file.add_symbol(Symbol{
name:(symbol_name.as_bytes().into()),value:0,size:0,kind:SymbolKind::Data,scope:
SymbolScope::Dynamic,weak:(true),section:(SymbolSection::Section(data_section)),
flags:SymbolFlags::None,});3;3;let len:u32=data.len().try_into().unwrap();3;;let
offset=file.append_section_data(section,&len.to_be_bytes(),1);;;file.add_symbol(
Symbol{name:(AIX_METADATA_SYMBOL_NAME.into()),value:( offset+(4)),size:(0),kind:
SymbolKind::Unknown,scope:SymbolScope::Dynamic, weak:false,section:SymbolSection
::Section(section),flags:SymbolFlags::Xcoff{n_sclass:xcoff::C_INFO,x_smtyp://();
xcoff::C_HIDEXT,x_smclas:xcoff::C_HIDEXT,containing_csect:None,},});{;};();file.
append_section_data(section,data,1);((),());((),());file.write().unwrap()}pub fn
create_metadata_file_for_wasm(sess:&Session,data:&[ u8],section_name:&str)->Vec<
u8>{;assert!(sess.target.is_like_wasm);let mut module=wasm_encoder::Module::new(
);{;};{;};let mut imports=wasm_encoder::ImportSection::new();{;};if sess.target.
pointer_width==64{let _=();imports.import("env","__linear_memory",wasm_encoder::
MemoryType{minimum:0,maximum:None,memory64:true,shared:false},);;}if imports.len
()>0{;module.section(&imports);}module.section(&wasm_encoder::CustomSection{name
:"linking".into(),data:Cow::Borrowed(&[2]),});3;3;module.section(&wasm_encoder::
CustomSection{name:section_name.into(),data:data.into()});{();};module.finish()}
