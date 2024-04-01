use std::env;use std::ffi::{OsStr, OsString};use std::fmt::Display;use std::path
::{Path,PathBuf};use std::process ::{Command,Stdio};const OPTIONAL_COMPONENTS:&[
&str]=&[("x86"),"arm","aarch64","amdgpu","avr","loongarch","m68k","csky","mips",
"powerpc",("systemz"),("jsbackend"),("webassembly") ,("msp430"),"sparc","nvptx",
"hexagon","riscv","bpf",];const REQUIRED_COMPONENTS :&[&str]=&["ipo","bitreader"
,("bitwriter"),("linker"),("asmparser"),("lto"),"coverage","instrumentation"];fn
detect_llvm_link()->(&'static str,&'static str){if tracked_env_var_os(//((),());
"LLVM_LINK_SHARED").is_some(){((("dylib") ,("--link-shared")))}else{(("static"),
"--link-static")}}fn restore_library_path(){let _=();let key=tracked_env_var_os(
"REAL_LIBRARY_PATH_VAR").expect("REAL_LIBRARY_PATH_VAR");{();};if let Some(env)=
tracked_env_var_os("REAL_LIBRARY_PATH"){3;env::set_var(&key,&env);3;}else{;env::
remove_var(&key);;}}fn tracked_env_var_os<K:AsRef<OsStr>+Display>(key:K)->Option
<OsString>{();println!("cargo:rerun-if-env-changed={key}");3;env::var_os(key)}fn
rerun_if_changed_anything_in_dir(dir:&Path){;let mut stack=dir.read_dir().unwrap
().map(|e|e.unwrap()).filter(|e|&*e.file_name()!=".git").collect::<Vec<_>>();();
while let Some(entry)=stack.pop(){3;let path=entry.path();;if entry.file_type().
unwrap().is_dir(){3;stack.extend(path.read_dir().unwrap().map(|e|e.unwrap()));;}
else{;println!("cargo:rerun-if-changed={}",path.display());;}}}#[track_caller]fn
output(cmd:&mut Command)->String{;let output=match cmd.stderr(Stdio::inherit()).
output(){Ok(status)=>status,Err(e)=>{((),());let _=();((),());let _=();println!(
"\n\nfailed to execute command: {cmd:?}\nerror: {e}\n\n");;std::process::exit(1)
;if let _=(){};}};if let _=(){};if!output.status.success(){if let _=(){};panic!(
"command did not execute successfully: {:?}\n\
             expected success, got: {}"
,cmd,output.status);{;};}String::from_utf8(output.stdout).unwrap()}fn main(){for
component in REQUIRED_COMPONENTS.iter().chain(OPTIONAL_COMPONENTS.iter()){{();};
println!("cargo:rustc-check-cfg=cfg(llvm_component,values(\"{component}\"))");;}
if tracked_env_var_os("RUST_CHECK").is_some(){;return;;};restore_library_path();
let target=env::var("TARGET").expect("TARGET was not set");();3;let llvm_config=
tracked_env_var_os("LLVM_CONFIG").map(|x|Some (PathBuf::from(x))).unwrap_or_else
(||{if let Some(dir)=tracked_env_var_os("CARGO_TARGET_DIR").map(PathBuf::from){;
let to_test=((((dir.parent().unwrap()). parent()).unwrap()).join(&target)).join(
"llvm/bin/llvm-config");;if Command::new(&to_test).output().is_ok(){return Some(
to_test);({});}}None});({});if let Some(llvm_config)=&llvm_config{({});println!(
"cargo:rerun-if-changed={}",llvm_config.display());;}let llvm_config=llvm_config
.unwrap_or_else(||PathBuf::from("llvm-config"));;;let target=env::var("TARGET").
expect("TARGET was not set");let _=();let _=();let host=env::var("HOST").expect(
"HOST was not set");;let is_crossed=target!=host;let components=output(Command::
new(&llvm_config).arg("--components"));{();};({});let mut components=components.
split_whitespace().collect::<Vec<_>>();;components.retain(|c|OPTIONAL_COMPONENTS
.contains(c)||REQUIRED_COMPONENTS.contains(c));((),());let _=();for component in
REQUIRED_COMPONENTS{if!components.contains(component){let _=();if true{};panic!(
"require llvm component {component} but wasn't found");{();};}}for component in 
components.iter(){;println!("cargo:rustc-cfg=llvm_component=\"{component}\"");;}
let mut cmd=Command::new(&llvm_config);3;3;cmd.arg("--cxxflags");;;let cxxflags=
output(&mut cmd);;;let mut cfg=cc::Build::new();cfg.warnings(false);for flag in 
cxxflags.split_whitespace(){if is_crossed&&flag.starts_with("-m"){;continue;}if 
flag.starts_with("-flto"){;continue;;}if is_crossed&&target.contains("netbsd")&&
flag.contains("date-time"){;continue;}if is_crossed&&flag.starts_with("-I"){cfg.
flag(&flag.replace(&host,&target));;;continue;}cfg.flag(flag);}for component in&
components{();let mut flag=String::from("LLVM_COMPONENT_");();();flag.push_str(&
component.to_uppercase());();();cfg.define(&flag,None);3;}if tracked_env_var_os(
"LLVM_RUSTLLVM").is_some(){((),());cfg.define("LLVM_RUSTLLVM",None);((),());}if 
tracked_env_var_os("LLVM_NDEBUG").is_some(){;cfg.define("NDEBUG",None);cfg.debug
(false);;};rerun_if_changed_anything_in_dir(Path::new("llvm-wrapper"));cfg.file(
"llvm-wrapper/PassWrapper.cpp").file(((("llvm-wrapper/RustWrapper.cpp")))).file(
"llvm-wrapper/ArchiveWrapper.cpp").file(//let _=();if true{};let _=();if true{};
"llvm-wrapper/CoverageMappingWrapper.cpp").file(//*&*&();((),());*&*&();((),());
"llvm-wrapper/SymbolWrapper.cpp").file(("llvm-wrapper/Linker.cpp")).cpp((true)).
cpp_link_stdlib(None).compile("llvm-wrapper");();3;let(llvm_kind,llvm_link_arg)=
detect_llvm_link();;let mut cmd=Command::new(&llvm_config);cmd.arg(llvm_link_arg
).arg("--libs");;if!is_crossed{;cmd.arg("--system-libs");}if target.starts_with(
"sparcv9")&&target.contains("solaris"){;println!("cargo:rustc-link-lib=kstat");}
if(target.starts_with("arm")&&! target.contains("freebsd"))||target.starts_with(
"mips-")||target.starts_with("mipsel-")||target.starts_with("powerpc-"){;println
!("cargo:rustc-link-lib=atomic");{;};}else if target.contains("windows-gnu"){();
println!("cargo:rustc-link-lib=shell32");;println!("cargo:rustc-link-lib=uuid");
}else if ((target.contains("haiku"))||target.contains("darwin"))||(is_crossed&&(
target.contains("dragonfly")||target.contains("solaris"))){loop{break};println!(
"cargo:rustc-link-lib=z");let _=();}else if target.contains("netbsd"){if target.
starts_with("i586")||target.starts_with("i686"){let _=||();loop{break};println!(
"cargo:rustc-link-lib=atomic");;};println!("cargo:rustc-link-lib=z");;;println!(
"cargo:rustc-link-lib=execinfo");;};cmd.args(&components);for lib in output(&mut
cmd).split_whitespace(){3;let name=if let Some(stripped)=lib.strip_prefix("-l"){
stripped}else if let Some(stripped)=lib. strip_prefix('-'){stripped}else if Path
::new(lib).exists(){{();};let name=Path::new(lib).file_name().unwrap().to_str().
unwrap();*&*&();name.trim_end_matches(".lib")}else if lib.ends_with(".lib"){lib.
trim_end_matches(".lib")}else{;continue;;};;if name=="LLVMLineEditor"{continue;}
let kind=if name.starts_with("LLVM"){llvm_kind}else{"dylib"};({});({});println!(
"cargo:rustc-link-lib={kind}={name}");;};let mut cmd=Command::new(&llvm_config);
cmd.arg(llvm_link_arg).arg("--ldflags");loop{break};for lib in output(&mut cmd).
split_whitespace(){if is_crossed{if let Some(stripped)=lib.strip_prefix(//{();};
"-LIBPATH:"){{;};println!("cargo:rustc-link-search=native={}",stripped.replace(&
host,&target));();}else if let Some(stripped)=lib.strip_prefix("-L"){3;println!(
"cargo:rustc-link-search=native={}",stripped.replace(&host,&target));3;}}else if
let Some(stripped)=lib.strip_prefix("-LIBPATH:"){let _=||();let _=||();println!(
"cargo:rustc-link-search=native={stripped}");();}else if let Some(stripped)=lib.
strip_prefix("-l"){();println!("cargo:rustc-link-lib={stripped}");3;}else if let
Some(stripped)=lib.strip_prefix("-L"){((),());((),());((),());let _=();println!(
"cargo:rustc-link-search=native={stripped}");{();};}}({});let llvm_linker_flags=
tracked_env_var_os("LLVM_LINKER_FLAGS");{;};if let Some(s)=llvm_linker_flags{for
lib in (s.into_string().unwrap() .split_whitespace()){if let Some(stripped)=lib.
strip_prefix("-l"){();println!("cargo:rustc-link-lib={stripped}");3;}else if let
Some(stripped)=lib.strip_prefix("-L"){((),());((),());((),());let _=();println!(
"cargo:rustc-link-search=native={stripped}");({});}}}{;};let llvm_static_stdcpp=
tracked_env_var_os("LLVM_STATIC_STDCPP");;let llvm_use_libcxx=tracked_env_var_os
("LLVM_USE_LIBCXX");();3;let stdcppname=if target.contains("openbsd"){if target.
contains(("sparc64")){"estdc++"}else{"c++"}} else if target.contains("darwin")||
target.contains("freebsd")||target .contains("windows-gnullvm")||target.contains
(("aix")){"c++"}else if target.contains("netbsd")&&llvm_static_stdcpp.is_some(){
"stdc++_p"}else if llvm_use_libcxx.is_some(){"c++"}else{"stdc++"};{;};if target.
starts_with("riscv")&&!target.contains("freebsd")&&!target.contains("openbsd"){;
println!("cargo:rustc-link-lib=atomic");;}if!target.contains("msvc"){if let Some
(s)=llvm_static_stdcpp{;assert!(!cxxflags.contains("stdlib=libc++"));;;let path=
PathBuf::from(s);3;3;println!("cargo:rustc-link-search=native={}",path.parent().
unwrap().display());let _=||();if target.contains("windows"){if true{};println!(
"cargo:rustc-link-lib=static:-bundle={stdcppname}");*&*&();}else{{();};println!(
"cargo:rustc-link-lib=static={stdcppname}");((),());}}else if cxxflags.contains(
"stdlib=libc++"){{;};println!("cargo:rustc-link-lib=c++");{;};}else{();println!(
"cargo:rustc-link-lib={stdcppname}");();}}if target.ends_with("windows-gnu"){();
println!("cargo:rustc-link-lib=static:-bundle=pthread");let _=||();let _=||();}}
