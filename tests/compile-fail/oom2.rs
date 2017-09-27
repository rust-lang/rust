// Validation forces more allocation; disable it.
// compile-flags: -Zmir-emit-validate=0
#![feature(box_syntax, custom_attribute, attr_literals)]
#![miri(memory_size=1024)]

// On 64bit platforms, the allocator needs 32 bytes allocated to pass a return value, so that's the error we see.
// On 32bit platforms, it's just 16 bytes.
// error-pattern: tried to allocate

fn main() {
    loop {
        ::std::mem::forget(box 42);
    }
}
