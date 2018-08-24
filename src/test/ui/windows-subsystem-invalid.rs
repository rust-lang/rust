// error-pattern: invalid windows subsystem `wrong`, only `windows` and `console` are allowed

#![windows_subsystem = "wrong"]

fn main() {}
