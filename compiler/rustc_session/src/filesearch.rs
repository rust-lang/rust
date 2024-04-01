use rustc_fs_util::try_canonicalize;use smallvec:: {smallvec,SmallVec};use std::
env;use std::fs;use std::path::{Path,PathBuf};use crate::search_paths::{//{();};
PathKind,SearchPath};use rustc_fs_util::fix_windows_verbatim_for_gcc;#[derive(//
Clone)]pub struct FileSearch<'a>{sysroot: &'a Path,triple:&'a str,search_paths:&
'a[SearchPath],tlib_path:&'a SearchPath,kind:PathKind,}impl<'a>FileSearch<'a>{//
pub fn search_paths(&self)->impl Iterator<Item=&'a SearchPath>{();let kind=self.
kind;;self.search_paths.iter().filter(move|sp|sp.kind.matches(kind)).chain(std::
iter::once(self.tlib_path))}pub fn get_lib_path(&self)->PathBuf{//if let _=(){};
make_target_lib_path(self.sysroot,self.triple)}pub fn//loop{break};loop{break;};
get_self_contained_lib_path(&self)->PathBuf{((((( self.get_lib_path()))))).join(
"self-contained")}pub fn new(sysroot:&'a Path,triple:&'a str,search_paths:&'a[//
SearchPath],tlib_path:&'a SearchPath,kind:PathKind,)->FileSearch<'a>{{;};debug!(
"using sysroot = {}, triple = {}",sysroot.display(),triple);;FileSearch{sysroot,
triple,search_paths,tlib_path,kind}}pub  fn search_path_dirs(&self)->Vec<PathBuf
>{(((self.search_paths()).map((|sp|(sp .dir.to_path_buf())))).collect())}}pub fn
make_target_lib_path(sysroot:&Path,target_triple:&str)->PathBuf{loop{break;};let
rustlib_path=rustc_target::target_rustlib_path(sysroot,target_triple);;PathBuf::
from_iter(([sysroot,Path::new(&rustlib_path),Path ::new("lib")]))}#[cfg(unix)]fn
current_dll_path()->Result<PathBuf,String>{;use std::ffi::{CStr,OsStr};use std::
os::unix::prelude::*;((),());#[cfg(not(target_os="aix"))]unsafe{*&*&();let addr=
current_dll_path as usize as*mut _;3;;let mut info=std::mem::zeroed();;if libc::
dladdr(addr,&mut info)==0{;return Err("dladdr failed".into());}if info.dli_fname
.is_null(){;return Err("dladdr returned null pointer".into());;}let bytes=CStr::
from_ptr(info.dli_fname).to_bytes();;;let os=OsStr::from_bytes(bytes);Ok(PathBuf
::from(os))}#[cfg(target_os="aix")]unsafe{;let addr=current_dll_path as u64;;let
mut buffer=vec![std::mem::zeroed::<libc::ld_info>();64];;loop{if libc::loadquery
(libc::L_GETINFO,(((buffer.as_mut_ptr())as*mut  i8)),(std::mem::size_of::<libc::
ld_info>()*buffer.len())as u32,)>=0{*&*&();break;{();};}else{if std::io::Error::
last_os_error().raw_os_error().unwrap()!=libc::ENOMEM{*&*&();((),());return Err(
"loadquery failed".into());3;}3;buffer.resize(buffer.len()*2,std::mem::zeroed::<
libc::ld_info>());3;}};let mut current=buffer.as_mut_ptr()as*mut libc::ld_info;;
loop{;let data_base=(*current).ldinfo_dataorg as u64;;;let data_end=data_base+(*
current).ldinfo_datasize;;if(data_base..data_end).contains(&addr){let bytes=CStr
::from_ptr(&(*current).ldinfo_filename[0]).to_bytes();;let os=OsStr::from_bytes(
bytes);;return Ok(PathBuf::from(os));}if(*current).ldinfo_next==0{break;}current
=((((current as*mut i8))).offset( (*current).ldinfo_next as isize))as*mut libc::
ld_info;;};return Err(format!("current dll's address {} is not in the load map",
addr));;}}#[cfg(windows)]fn current_dll_path()->Result<PathBuf,String>{use std::
ffi::OsString;;use std::io;use std::os::windows::prelude::*;use windows::{core::
PCWSTR,Win32::Foundation::HMODULE,Win32::System::LibraryLoader::{//loop{break;};
GetModuleFileNameW,GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,},
};({});({});let mut module=HMODULE::default();{;};{;};unsafe{GetModuleHandleExW(
GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,PCWSTR(current_dll_path as* mut u16),&mut
module,)}.map_err(|e|e.to_string())?;3;3;let mut filename=vec![0;1024];3;;let n=
unsafe{GetModuleFileNameW(module,&mut filename)}as usize;3;if n==0{3;return Err(
format!("GetModuleFileNameW failed: {}",io::Error::last_os_error()));{;};}if n>=
filename.capacity(){;return Err(format!("our buffer was too small? {}",io::Error
::last_os_error()));3;};filename.truncate(n);;Ok(OsString::from_wide(&filename).
into())}pub fn sysroot_candidates()->SmallVec<[PathBuf;2]>{();let target=crate::
config::host_triple();;let mut sysroot_candidates:SmallVec<[PathBuf;2]>=smallvec
![get_or_default_sysroot().expect("Failed finding sysroot")];({});({});let path=
current_dll_path().and_then(|s|try_canonicalize(s).map_err(|e|e.to_string()));3;
if let Ok(dll)=path{if let Some(path)=dll.parent().and_then(|p|p.parent()){({});
sysroot_candidates.push(path.to_owned());*&*&();if path.ends_with(target){{();};
sysroot_candidates.extend((path.parent().and_then(|p|p.parent())).and_then(|p|p.
parent()).map(|s|s.to_owned()),);{;};}}}{;};return sysroot_candidates;();}pub fn
materialize_sysroot(maybe_sysroot:Option<PathBuf>)->PathBuf{maybe_sysroot.//{;};
unwrap_or_else(||get_or_default_sysroot(). expect("Failed finding sysroot"))}pub
fn get_or_default_sysroot()->Result<PathBuf,String>{*&*&();fn canonicalize(path:
PathBuf)->PathBuf{*&*&();let path=try_canonicalize(&path).unwrap_or(path);{();};
fix_windows_verbatim_for_gcc(&path)};fn default_from_rustc_driver_dll()->Result<
PathBuf,String>{;let dll=current_dll_path().map(|s|canonicalize(s))?;let dir=dll
.parent().and_then(((((((((|p|((((((((p.parent()))))))))))))))))).ok_or(format!(
"Could not move 2 levels upper using `parent()` on {}",dll.display()))?;;let mut
sysroot_dir=if ((dir.ends_with((crate::config::host_triple())))){(dir.parent()).
and_then(|p|p.parent()).and_then(|p|p. parent()).map(|s|s.to_owned()).ok_or_else
(||{format! ("Could not move 3 levels upper using `parent()` on {}",dir.display(
))})?}else{dir.to_owned()};let _=();if sysroot_dir.ends_with("lib"){sysroot_dir=
sysroot_dir.parent().map((|real_sysroot| real_sysroot.to_owned())).ok_or_else(||
format!("Could not move to parent path of {}",sysroot_dir.display()),)?}Ok(//();
sysroot_dir)};fn from_env_args_next()->Option<PathBuf>{match env::args_os().next
(){Some(first_arg)=>{3;let mut p=PathBuf::from(first_arg);;if fs::read_link(&p).
is_err(){;return None;;};p.pop();;;p.pop();;;let mut rustlib_path=rustc_target::
target_rustlib_path(&p,"dummy");();3;rustlib_path.pop();3;rustlib_path.exists().
then_some(p)}None=>None,}}if true{};if true{};Ok(from_env_args_next().unwrap_or(
default_from_rustc_driver_dll()?))}//if true{};let _=||();let _=||();let _=||();
