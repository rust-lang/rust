// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdbg-check:$1 = {data_ptr = [...], length = 4}
// gdbr-check:$1 = &unsized::Foo<[u8]> {data_ptr: [...], length: 4}

// gdb-command:print b
// gdbg-check:$2 = {data_ptr = [...], length = 4}
// gdbr-check:$2 = &unsized::Foo<unsized::Foo<[u8]>> {data_ptr: [...], length: 4}

// gdb-command:print c
// gdbg-check:$3 = {pointer = [...], vtable = [...]}
// gdbr-check:$3 = &unsized::Foo<dyn core::fmt::Debug> {pointer: [...], vtable: [...]}

// gdb-command:print tuple_slice
// gdbg-check:$4 = {data_ptr = [...], length = 2}
// gdbr-check:$4 = &(i32, i32, [i32]) {data_ptr: [...], length: 2}

// gdb-command:print tuple_dyn
// gdbg-check:$5 = {pointer = [...], vtable = [...]}
// gdbr-check:$5 = &(i32, i32, dyn core::fmt::Debug) {pointer: [...], vtable: [...]}

// === CDB TESTS ===================================================================================

// cdb-command: g
// cdb-command:dx a
// cdb-check:a                [Type: ref$<unsized::Foo<slice$<u8> > >]
// cdb-check:    [+0x000] data_ptr         : 0x[...] [Type: unsized::Foo<slice$<u8> > *]
// cdb-check:    [...] length           : 0x4 [Type: unsigned [...]int[...]

// cdb-command:dx b
// cdb-check:b                [Type: ref$<unsized::Foo<unsized::Foo<slice$<u8> > > >]
// cdb-check:    [+0x000] data_ptr         : 0x[...] [Type: unsized::Foo<unsized::Foo<slice$<u8> > > *]
// cdb-check:    [...] length           : 0x4 [Type: unsigned [...]int[...]

// cdb-command:dx c
// cdb-check:c                [Type: ref$<unsized::Foo<dyn$<core::fmt::Debug> > >]
// cdb-check:    [+0x000] pointer          : 0x[...] [Type: unsized::Foo<dyn$<core::fmt::Debug> > *]
// cdb-check:    [...] vtable           : 0x[...] [Type: unsigned [...]int[...] (*)[3]]

// cdb-command:dx tuple_slice
// cdb-check:tuple_slice      [Type: ref$<tuple$<i32,i32,slice$<i32> > >]
// cdb-check:    [+0x000] data_ptr         : 0x[...] [Type: tuple$<i32,i32,slice$<i32> > *]
// cdb-check:    [...] length           : 0x2 [Type: unsigned [...]int[...]

// cdb-command:dx tuple_dyn
// cdb-check:tuple_dyn        [Type: ref$<tuple$<i32,i32,dyn$<core::fmt::Debug> > >]
// cdb-check:    [+0x000] pointer          : 0x[...] [Type: tuple$<i32,i32,dyn$<core::fmt::Debug> > *]
// cdb-check:    [...] vtable           : 0x[...] [Type: unsigned [...]int[...] (*)[3]]

#![feature(unsized_tuple_coercion)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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

    // Also check unsized tuples
    let tuple_slice: &(i32, i32, [i32]) = &(0, 1, [2, 3]);
    let tuple_dyn: &(i32, i32, dyn std::fmt::Debug) = &(0, 1, &3u64);

    zzz(); // #break
}

fn zzz() {
    ()
}
