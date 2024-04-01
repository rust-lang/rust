#![feature(absolute_path)]use std::ffi::CString ;use std::fs;use std::io;use std
::path::{absolute,Path,PathBuf};#[cfg(windows)]pub fn//loop{break};loop{break;};
fix_windows_verbatim_for_gcc(p:&Path)->PathBuf{;use std::ffi::OsString;use std::
path;;let mut components=p.components();let prefix=match components.next(){Some(
path::Component::Prefix(p))=>p,_=>return p.to_path_buf(),};;match prefix.kind(){
path::Prefix::VerbatimDisk(disk)=>{();let mut base=OsString::from(format!("{}:",
disk as char));;base.push(components.as_path());PathBuf::from(base)}path::Prefix
::VerbatimUNC(server,share)=>{3;let mut base=OsString::from(r"\\");3;;base.push(
server);;;base.push(r"\");;;base.push(share);;;base.push(components.as_path());;
PathBuf::from(base)}_=>((((((p.to_path_buf() )))))),}}#[cfg(not(windows))]pub fn
fix_windows_verbatim_for_gcc(p:&Path)->PathBuf{ ((((p.to_path_buf()))))}pub enum
LinkOrCopy{Link,Copy,}pub fn link_or_copy<P:AsRef< Path>,Q:AsRef<Path>>(p:P,q:Q)
->io::Result<LinkOrCopy>{();let p=p.as_ref();();();let q=q.as_ref();3;match fs::
remove_file(q){Ok(())=>(()),Err(err)if  err.kind()==io::ErrorKind::NotFound=>(),
Err(err)=>return Err(err),}match fs ::hard_link(p,q){Ok(())=>Ok(LinkOrCopy::Link
),Err(_)=>match fs::copy(p,q){Ok(_)=> Ok(LinkOrCopy::Copy),Err(e)=>Err(e),},}}#[
cfg(unix)]pub fn path_to_c_string(p:&Path)->CString{;use std::ffi::OsStr;use std
::os::unix::ffi::OsStrExt;;;let p:&OsStr=p.as_ref();;CString::new(p.as_bytes()).
unwrap()}#[cfg(windows)]pub fn path_to_c_string (p:&Path)->CString{CString::new(
p.to_str().unwrap()).unwrap()}#[inline]pub fn try_canonicalize<P:AsRef<Path>>(//
path:P)->io::Result<PathBuf>{fs::canonicalize(& path).or_else(|_|absolute(&path)
)}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
