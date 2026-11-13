//@ compile-flags:-g
//@ disable-gdb-pretty-printers
//@ ignore-gdb-version: 13.1 - 99.0
//@ ignore-backends: gcc
// ^ https://sourceware.org/bugzilla/show_bug.cgi?id=30330

// === GDB TESTS ===================================================================================

//@ gdb-command:run

//@ gdb-command:print a
//@ gdb-check:$1 = &unsized::Foo<[u8]> {data_ptr: [...], length: 4}

//@ gdb-command:print b
//@ gdb-check:$2 = &unsized::Foo<unsized::Foo<[u8]>> {data_ptr: [...], length: 4}

//@ gdb-command:print c
//@ gdb-check:$3 = &unsized::Foo<dyn core::fmt::Debug> {pointer: [...], vtable: [...]}

//@ gdb-command:print _box
//@ gdb-check:$4 = alloc::boxed::Box<unsized::Foo<dyn core::fmt::Debug>, alloc::alloc::Global> {pointer: [...], vtable: [...]}

// === CDB TESTS ===================================================================================

// NB: `c` and `_box` format differences between 10.0.26100.0 and 10.0.28000.2957.
//@ min-cdb-version: 10.0.28000.2957

//@ cdb-command: g
//@ cdb-command:dx a
//@ cdb-check:a                [Type: ref$<unsized::Foo<slice2$<u8> > >]
//@ cdb-check:    [+0x000] data_ptr         : 0x[...] [Type: unsized::Foo<slice2$<u8> > *]
//@ cdb-check:    [...] length           : 0x4 [Type: unsigned [...]int[...]

//@ cdb-command:dx b
//@ cdb-check:b                [Type: ref$<unsized::Foo<unsized::Foo<slice2$<u8> > > >]
//@ cdb-check:    [+0x000] data_ptr         : 0x[...] [Type: unsized::Foo<unsized::Foo<slice2$<u8> > > *]
//@ cdb-check:    [...] length           : 0x4 [Type: unsigned [...]int[...]

//@ cdb-command:dx c
//@ cdb-check:c                : 7 [Type: i32 (derived from ref$<unsized::Foo<dyn$<core::fmt::Debug> > >)]

//@ cdb-command:dx _box
//@ cdb-check:_box             : 8 [Type: i32 (derived from alloc::boxed::Box<unsized::Foo<dyn$<core::fmt::Debug> >,alloc::alloc::Global>)]

struct Foo<T: ?Sized> {
    value: T,
}

fn main() {
    let foo: Foo<Foo<[u8; 4]>> = Foo { value: Foo { value: *b"abc\0" } };

    // We expect `a`, `b`, and `c` to all be fat pointers.
    // `a` and `b` should be slice-like and thus have a `data_ptr` and `length` field.
    // `c` should be trait-object-like and thus have a `pointer` and `vtable` field.
    let a: &Foo<[u8]> = &foo.value;
    let b: &Foo<Foo<[u8]>> = &foo;
    let c: &Foo<dyn std::fmt::Debug> = &Foo { value: 7i32 };
    let _box: Box<Foo<dyn std::fmt::Debug>> = Box::new(Foo { value: 8i32 });

    zzz(); // #break
}

fn zzz() {
    ()
}
