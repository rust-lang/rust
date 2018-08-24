// error-pattern: mod statements in non-mod.rs files are unstable

// This is not a directory owner since the file name is not "mod.rs".
#[path = "mod_file_not_owning_aux1.rs"]
mod foo;
