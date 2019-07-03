// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

const INVALID_ENUM : u32 = 0;
const INVALID_VALUE : u32 = 1;

fn gl_err_str(err: u32) -> String
{
  match err
  {
    INVALID_ENUM => { "Invalid enum".to_string() },
    INVALID_VALUE => { "Invalid value".to_string() },
    _ => { "Unknown error".to_string() }
  }
}

pub fn main() {}
