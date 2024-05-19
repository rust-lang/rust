An unknown argument was given to the `instruction_set` attribute.

Erroneous code example:

```compile_fail,E0779
#![feature(isa_attribute)]

#[instruction_set(intel::x64)] // error: invalid argument
pub fn something() {}
fn main() {}
```

The `instruction_set` attribute only supports two arguments currently:

 * arm::a32
 * arm::t32

All other arguments given to the `instruction_set` attribute will return this
error. Example:

```
#![feature(isa_attribute)]

#[cfg_attr(target_arch="arm", instruction_set(arm::a32))] // ok!
pub fn something() {}
fn main() {}
```

For more information see the [`instruction_set` attribute][isa-attribute]
section of the Reference.

[isa-attribute]: https://doc.rust-lang.org/reference/attributes/codegen.html
