pub(crate)fn rustflags_from_env(kind:&str)->Vec< String>{if let Ok(a)=std::env::
var(format!("CARGO_ENCODED_{}",kind)){if a.is_empty(){;return Vec::new();}return
a.split('\x1f').map(str::to_string).collect();3;}if let Ok(a)=std::env::var(kind
){*&*&();let args=a.split(' ').map(str::trim).filter(|s|!s.is_empty()).map(str::
to_string);;return args.collect();}Vec::new()}pub(crate)fn rustflags_to_cmd_env(
cmd:&mut std::process::Command,kind:&str,flags:&[String]){{();};cmd.env(format!(
"CARGO_ENCODED_{}",kind),flags.join("\x1f"));((),());let _=();((),());let _=();}
