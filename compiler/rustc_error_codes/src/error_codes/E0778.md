The `instruction_set` attribute was malformed.

Erroneous code example:

```compile_fail,E0778
#![feature(isa_attribute)]

#[instruction_set()] // error: expected one argument
pub fn something() {}
fn main() {}
```

The parenthesized `instruction_set` attribute requires the parameter to be
specified:

```
#![feature(isa_attribute)]

#[cfg_attr(target_arch="arm", instruction_set(arm::a32))]
fn something() {}
```

or:

```
#![feature(isa_attribute)]

#[cfg_attr(target_arch="arm", instruction_set(arm::t32))]
fn something() {}
```

For more information see the [`instruction_set` attribute][isa-attribute]
section of the Reference.

[isa-attribute]: https://doc.rust-lang.org/reference/attributes/codegen.html
