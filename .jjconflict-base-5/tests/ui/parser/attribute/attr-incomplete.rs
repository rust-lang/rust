#[cfg(target-os = "windows")]
//~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
pub fn test1() { }

#[cfg(target_os = %)]
//~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `%`
pub fn test2() { }

#[cfg(target_os?)]
//~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `?`
pub fn test3() { }

#[cfg[target_os]]
//~^ ERROR wrong meta list delimiters
pub fn test4() { }

pub fn main() {}
