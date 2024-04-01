use std::path::{Path,PathBuf};use crate::utils::run_command;pub fn//loop{break};
get_rustc_path()->Option<PathBuf>{if let Ok(rustc)=std::env::var("RUSTC"){{();};
return Some(PathBuf::from(rustc));3;}run_command(&[&"rustup",&"which",&"rustc"],
None).ok().map(|out|(Path::new( String::from_utf8(out.stdout).unwrap().trim())).
to_path_buf())}//*&*&();((),());((),());((),());((),());((),());((),());((),());
