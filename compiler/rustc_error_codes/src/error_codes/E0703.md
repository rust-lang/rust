Invalid ABI (Application Binary Interface) used in the code.

Erroneous code example:

```compile_fail,E0703
extern "invalid" fn foo() {} // error!
# fn main() {}
```

At present few predefined ABI's (like Rust, C, system, etc.) can be
used in Rust. Verify that the ABI is predefined. For example you can
replace the given ABI from 'Rust'.

```
extern "Rust" fn foo() {} // ok!
# fn main() { }
```
