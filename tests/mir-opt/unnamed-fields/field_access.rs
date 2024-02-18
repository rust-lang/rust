// skip-filecheck
// EMIT_MIR field_access.foo.SimplifyCfg-initial.after.mir
// EMIT_MIR field_access.bar.SimplifyCfg-initial.after.mir

#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[repr(C)]
struct Foo {
    a: u8,
    _: struct {
        b: i8,
        c: bool,
    },
    _: struct {
        _: struct {
            d: [u8; 1],
        }
    }
}

#[repr(C)]
union Bar {
    a: u8,
    _: union {
        b: i8,
        c: bool,
    },
    _: union {
        _: union {
            d: [u8; 1],
        }
    }
}


fn access<T>(_: T) {}

fn foo(foo: Foo) {
    access(foo.a);
    access(foo.b);
    access(foo.c);
    access(foo.d);
}

fn bar(bar: Bar) {
    unsafe {
        access(bar.a);
        access(bar.b);
        access(bar.c);
        access(bar.d);
    }
}


fn main() {}
