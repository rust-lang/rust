// run-pass
#![allow(unused_variables)]
use std::ops::Deref;

fn main() {
    if env_var("FOOBAR").as_ref().map(Deref::deref).ok() == Some("yes") {
        panic!()
    }

    let env_home: Result<String, ()> = Ok("foo-bar-baz".to_string());
    let env_home = env_home.as_ref().map(Deref::deref).ok();

    if env_home == Some("") { panic!() }
}

#[inline(never)]
fn env_var(s: &str) -> Result<String, VarError> {
    Err(VarError::NotPresent)
}

pub enum VarError {
    NotPresent,
    NotUnicode(String),
}
