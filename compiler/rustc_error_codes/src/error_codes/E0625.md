A compile-time const variable is referring to a thread-local static variable.

Erroneous code example:

```compile_fail,E0625
#![feature(thread_local)]

#[thread_local]
static X: usize = 12;

const Y: usize = 2 * X;
```

Static and const variables can refer to other const variables but a const
variable cannot refer to a thread-local static variable. In this example,
`Y` cannot refer to `X`. To fix this, the value can be extracted as a const
and then used:

```
#![feature(thread_local)]

const C: usize = 12;

#[thread_local]
static X: usize = C;

const Y: usize = 2 * C;
```
