// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdb-check:$1 = 1
// gdb-command:print b
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:print a
// gdb-check:$3 = 2
// gdb-command:print b
// gdb-check:$4 = 3
// gdb-command:print c
// gdb-check:$5 = 4
// gdb-command:continue

// gdb-command:print a
// gdb-check:$6 = 5
// gdb-command:print b
// gdbg-check:$7 = {__0 = 6, __1 = 7}
// gdbr-check:$7 = (6, 7)
// gdb-command:continue

// gdb-command:print h
// gdb-check:$8 = 8
// gdb-command:print i
// gdbg-check:$9 = {a = 9, b = 10}
// gdbr-check:$9 = destructured_fn_argument::Struct {a: 9, b: 10}
// gdb-command:print j
// gdb-check:$10 = 11
// gdb-command:continue

// gdb-command:print k
// gdb-check:$11 = 12
// gdb-command:print l
// gdb-check:$12 = 13
// gdb-command:continue

// gdb-command:print m
// gdb-check:$13 = 14
// gdb-command:print n
// gdb-check:$14 = 16
// gdb-command:continue

// gdb-command:print o
// gdb-check:$15 = 18
// gdb-command:continue

// gdb-command:print p
// gdb-check:$16 = 19
// gdb-command:print q
// gdb-check:$17 = 20
// gdb-command:print r
// gdbg-check:$18 = {a = 21, b = 22}
// gdbr-check:$18 = destructured_fn_argument::Struct {a: 21, b: 22}
// gdb-command:continue

// gdb-command:print s
// gdb-check:$19 = 24
// gdb-command:print t
// gdb-check:$20 = 23
// gdb-command:continue

// gdb-command:print u
// gdb-check:$21 = 25
// gdb-command:print v
// gdb-check:$22 = 26
// gdb-command:print w
// gdb-check:$23 = 27
// gdb-command:print x
// gdb-check:$24 = 28
// gdb-command:print y
// gdb-check:$25 = 29
// gdb-command:print z
// gdb-check:$26 = 30
// gdb-command:print ae
// gdb-check:$27 = 31
// gdb-command:print oe
// gdb-check:$28 = 32
// gdb-command:print ue
// gdb-check:$29 = 33
// gdb-command:continue

// gdb-command:print aa
// gdbg-check:$30 = {__0 = 34, __1 = 35}
// gdbr-check:$30 = (34, 35)
// gdb-command:continue

// gdb-command:print bb
// gdbg-check:$31 = {__0 = 36, __1 = 37}
// gdbr-check:$31 = (36, 37)
// gdb-command:continue

// gdb-command:print cc
// gdb-check:$32 = 38
// gdb-command:continue

// gdb-command:print dd
// gdbg-check:$33 = {__0 = 40, __1 = 41, __2 = 42}
// gdbr-check:$33 = (40, 41, 42)
// gdb-command:continue

// gdb-command:print *ee
// gdbg-check:$34 = {__0 = 43, __1 = 44, __2 = 45}
// gdbr-check:$34 = (43, 44, 45)
// gdb-command:continue

// gdb-command:print *ff
// gdb-check:$35 = 46
// gdb-command:print gg
// gdbg-check:$36 = {__0 = 47, __1 = 48}
// gdbr-check:$36 = (47, 48)
// gdb-command:continue

// gdb-command:print *hh
// gdb-check:$37 = 50
// gdb-command:continue

// gdb-command:print ii
// gdb-check:$38 = 51
// gdb-command:continue

// gdb-command:print *jj
// gdb-check:$39 = 52
// gdb-command:continue

// gdb-command:print kk
// gdb-check:$40 = 53
// gdb-command:print ll
// gdb-check:$41 = 54
// gdb-command:continue

// gdb-command:print mm
// gdb-check:$42 = 55
// gdb-command:print *nn
// gdb-check:$43 = 56
// gdb-command:continue

// gdb-command:print oo
// gdb-check:$44 = 57
// gdb-command:print pp
// gdb-check:$45 = 58
// gdb-command:print qq
// gdb-check:$46 = 59
// gdb-command:continue

// gdb-command:print rr
// gdb-check:$47 = 60
// gdb-command:print ss
// gdb-check:$48 = 61
// gdb-command:print tt
// gdb-check:$49 = 62
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print a
// lldb-check:[...]$0 = 1
// lldb-command:print b
// lldb-check:[...]$1 = false
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$2 = 2
// lldb-command:print b
// lldb-check:[...]$3 = 3
// lldb-command:print c
// lldb-check:[...]$4 = 4
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$5 = 5
// lldb-command:print b
// lldb-check:[...]$6 = (6, 7)
// lldb-command:continue

// lldb-command:print h
// lldb-check:[...]$7 = 8
// lldb-command:print i
// lldb-check:[...]$8 = Struct { a: 9, b: 10 }
// lldb-command:print j
// lldb-check:[...]$9 = 11
// lldb-command:continue

// lldb-command:print k
// lldb-check:[...]$10 = 12
// lldb-command:print l
// lldb-check:[...]$11 = 13
// lldb-command:continue

// lldb-command:print m
// lldb-check:[...]$12 = 14
// lldb-command:print n
// lldb-check:[...]$13 = 16
// lldb-command:continue

// lldb-command:print o
// lldb-check:[...]$14 = 18
// lldb-command:continue

// lldb-command:print p
// lldb-check:[...]$15 = 19
// lldb-command:print q
// lldb-check:[...]$16 = 20
// lldb-command:print r
// lldb-check:[...]$17 = Struct { a: 21, b: 22 }
// lldb-command:continue

// lldb-command:print s
// lldb-check:[...]$18 = 24
// lldb-command:print t
// lldb-check:[...]$19 = 23
// lldb-command:continue

// lldb-command:print u
// lldb-check:[...]$20 = 25
// lldb-command:print v
// lldb-check:[...]$21 = 26
// lldb-command:print w
// lldb-check:[...]$22 = 27
// lldb-command:print x
// lldb-check:[...]$23 = 28
// lldb-command:print y
// lldb-check:[...]$24 = 29
// lldb-command:print z
// lldb-check:[...]$25 = 30
// lldb-command:print ae
// lldb-check:[...]$26 = 31
// lldb-command:print oe
// lldb-check:[...]$27 = 32
// lldb-command:print ue
// lldb-check:[...]$28 = 33
// lldb-command:continue

// lldb-command:print aa
// lldb-check:[...]$29 = (34, 35)
// lldb-command:continue

// lldb-command:print bb
// lldb-check:[...]$30 = (36, 37)
// lldb-command:continue

// lldb-command:print cc
// lldb-check:[...]$31 = 38
// lldb-command:continue

// lldb-command:print dd
// lldb-check:[...]$32 = (40, 41, 42)
// lldb-command:continue

// lldb-command:print *ee
// lldb-check:[...]$33 = (43, 44, 45)
// lldb-command:continue

// lldb-command:print *ff
// lldb-check:[...]$34 = 46
// lldb-command:print gg
// lldb-check:[...]$35 = (47, 48)
// lldb-command:continue

// lldb-command:print *hh
// lldb-check:[...]$36 = 50
// lldb-command:continue

// lldb-command:print ii
// lldb-check:[...]$37 = 51
// lldb-command:continue

// lldb-command:print *jj
// lldb-check:[...]$38 = 52
// lldb-command:continue

// lldb-command:print kk
// lldb-check:[...]$39 = 53
// lldb-command:print ll
// lldb-check:[...]$40 = 54
// lldb-command:continue

// lldb-command:print mm
// lldb-check:[...]$41 = 55
// lldb-command:print *nn
// lldb-check:[...]$42 = 56
// lldb-command:continue

// lldb-command:print oo
// lldb-check:[...]$43 = 57
// lldb-command:print pp
// lldb-check:[...]$44 = 58
// lldb-command:print qq
// lldb-check:[...]$45 = 59
// lldb-command:continue

// lldb-command:print rr
// lldb-check:[...]$46 = 60
// lldb-command:print ss
// lldb-check:[...]$47 = 61
// lldb-command:print tt
// lldb-check:[...]$48 = 62
// lldb-command:continue

#![allow(unused_variables)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Univariant::Unit;

struct Struct {
    a: i64,
    b: i32
}

enum Univariant {
    Unit(i32)
}

struct TupleStruct (f64, isize);


fn simple_tuple((a, b): (isize, bool)) {
    zzz(); // #break
}

fn nested_tuple((a, (b, c)): (isize, (u16, u16))) {
    zzz(); // #break
}

fn destructure_only_first_level((a, b): (isize, (u32, u32))) {
    zzz(); // #break
}

fn struct_as_tuple_element((h, i, j): (i16, Struct, i16)) {
    zzz(); // #break
}

fn struct_pattern(Struct { a: k, b: l }: Struct) {
    zzz(); // #break
}

fn ignored_tuple_element((m, _, n): (isize, u16, i32)) {
    zzz(); // #break
}

fn ignored_struct_field(Struct { b: o, .. }: Struct) {
    zzz(); // #break
}

fn one_struct_destructured_one_not((Struct { a: p, b: q }, r): (Struct, Struct)) {
    zzz(); // #break
}

fn different_order_of_struct_fields(Struct { b: s, a: t }: Struct ) {
    zzz(); // #break
}

fn complex_nesting(((u,   v  ), ((w,   (x,   Struct { a: y, b: z})), Struct { a: ae, b: oe }), ue ):
                   ((i16, i32), ((i64, (i32, Struct,             )), Struct                 ), u16))
{
    zzz(); // #break
}

fn managed_box(&aa: &(isize, isize)) {
    zzz(); // #break
}

fn borrowed_pointer(&bb: &(isize, isize)) {
    zzz(); // #break
}

fn contained_borrowed_pointer((&cc, _): (&isize, isize)) {
    zzz(); // #break
}

fn unique_pointer(box dd: Box<(isize, isize, isize)>) {
    zzz(); // #break
}

fn ref_binding(ref ee: (isize, isize, isize)) {
    zzz(); // #break
}

fn ref_binding_in_tuple((ref ff, gg): (isize, (isize, isize))) {
    zzz(); // #break
}

fn ref_binding_in_struct(Struct { b: ref hh, .. }: Struct) {
    zzz(); // #break
}

fn univariant_enum(Unit(ii): Univariant) {
    zzz(); // #break
}

fn univariant_enum_with_ref_binding(Unit(ref jj): Univariant) {
    zzz(); // #break
}

fn tuple_struct(TupleStruct(kk, ll): TupleStruct) {
    zzz(); // #break
}

fn tuple_struct_with_ref_binding(TupleStruct(mm, ref nn): TupleStruct) {
    zzz(); // #break
}

fn multiple_arguments((oo, pp): (isize, isize), qq : isize) {
    zzz(); // #break
}

fn main() {
    simple_tuple((1, false));
    nested_tuple((2, (3, 4)));
    destructure_only_first_level((5, (6, 7)));
    struct_as_tuple_element((8, Struct { a: 9, b: 10 }, 11));
    struct_pattern(Struct { a: 12, b: 13 });
    ignored_tuple_element((14, 15, 16));
    ignored_struct_field(Struct { a: 17, b: 18 });
    one_struct_destructured_one_not((Struct { a: 19, b: 20 }, Struct { a: 21, b: 22 }));
    different_order_of_struct_fields(Struct { a: 23, b: 24 });
    complex_nesting(((25, 26), ((27, (28, Struct { a: 29, b: 30})), Struct { a: 31, b: 32 }), 33));
    managed_box(&(34, 35));
    borrowed_pointer(&(36, 37));
    contained_borrowed_pointer((&38, 39));
    unique_pointer(box (40, 41, 42));
    ref_binding((43, 44, 45));
    ref_binding_in_tuple((46, (47, 48)));
    ref_binding_in_struct(Struct { a: 49, b: 50 });
    univariant_enum(Unit(51));
    univariant_enum_with_ref_binding(Unit(52));
    tuple_struct(TupleStruct(53.0, 54));
    tuple_struct_with_ref_binding(TupleStruct(55.0, 56));
    multiple_arguments((57, 58), 59);

    fn nested_function(rr: isize, (ss, tt): (isize, isize)) {
        zzz(); // #break
    }

    nested_function(60, (61, 62));
}

fn zzz() { () }
