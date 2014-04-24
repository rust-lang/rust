// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print a
// gdb-check:$1 = 1
// gdb-command:print b
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$3 = 2
// gdb-command:print b
// gdb-check:$4 = 3
// gdb-command:print c
// gdb-check:$5 = 4
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$6 = 5
// gdb-command:print b
// gdb-check:$7 = {6, 7}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print h
// gdb-check:$8 = 8
// gdb-command:print i
// gdb-check:$9 = {a = 9, b = 10}
// gdb-command:print j
// gdb-check:$10 = 11
// gdb-command:continue

// gdb-command:finish
// gdb-command:print k
// gdb-check:$11 = 12
// gdb-command:print l
// gdb-check:$12 = 13
// gdb-command:continue

// gdb-command:finish
// gdb-command:print m
// gdb-check:$13 = 14
// gdb-command:print n
// gdb-check:$14 = 16
// gdb-command:continue

// gdb-command:finish
// gdb-command:print o
// gdb-check:$15 = 18
// gdb-command:continue

// gdb-command:finish
// gdb-command:print p
// gdb-check:$16 = 19
// gdb-command:print q
// gdb-check:$17 = 20
// gdb-command:print r
// gdb-check:$18 = {a = 21, b = 22}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print s
// gdb-check:$19 = 24
// gdb-command:print t
// gdb-check:$20 = 23
// gdb-command:continue

// gdb-command:finish
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

// gdb-command:finish
// gdb-command:print aa
// gdb-check:$30 = {34, 35}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print bb
// gdb-check:$31 = {36, 37}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print cc
// gdb-check:$32 = 38
// gdb-command:continue

// gdb-command:finish
// gdb-command:print dd
// gdb-check:$33 = {40, 41, 42}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print *ee
// gdb-check:$34 = {43, 44, 45}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print *ff
// gdb-check:$35 = 46
// gdb-command:print gg
// gdb-check:$36 = {47, 48}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print *hh
// gdb-check:$37 = 50
// gdb-command:continue

// gdb-command:finish
// gdb-command:print ii
// gdb-check:$38 = 51
// gdb-command:continue

// gdb-command:finish
// gdb-command:print *jj
// gdb-check:$39 = 52
// gdb-command:continue

// gdb-command:finish
// gdb-command:print kk
// gdb-check:$40 = 53
// gdb-command:print ll
// gdb-check:$41 = 54
// gdb-command:continue

// gdb-command:finish
// gdb-command:print mm
// gdb-check:$42 = 55
// gdb-command:print *nn
// gdb-check:$43 = 56
// gdb-command:continue

// gdb-command:finish
// gdb-command:print oo
// gdb-check:$44 = 57
// gdb-command:print pp
// gdb-check:$45 = 58
// gdb-command:print qq
// gdb-check:$46 = 59
// gdb-command:continue

// gdb-command:finish
// gdb-command:print rr
// gdb-check:$47 = 60
// gdb-command:print ss
// gdb-check:$48 = 61
// gdb-command:print tt
// gdb-check:$49 = 62
// gdb-command:continue

#![allow(unused_variable)]


struct Struct {
    a: i64,
    b: i32
}

enum Univariant {
    Unit(i32)
}

struct TupleStruct (f64, int);


fn simple_tuple((a, b): (int, bool)) {
    zzz();
}

fn nested_tuple((a, (b, c)): (int, (u16, u16))) {
    zzz();
}

fn destructure_only_first_level((a, b): (int, (u32, u32))) {
    zzz();
}

fn struct_as_tuple_element((h, i, j): (i16, Struct, i16)) {
    zzz();
}

fn struct_pattern(Struct { a: k, b: l }: Struct) {
    zzz();
}

fn ignored_tuple_element((m, _, n): (int, u16, i32)) {
    zzz();
}

fn ignored_struct_field(Struct { b: o, .. }: Struct) {
    zzz();
}

fn one_struct_destructured_one_not((Struct { a: p, b: q }, r): (Struct, Struct)) {
    zzz();
}

fn different_order_of_struct_fields(Struct { b: s, a: t }: Struct ) {
    zzz();
}

fn complex_nesting(((u,   v  ), ((w,   (x,   Struct { a: y, b: z})), Struct { a: ae, b: oe }), ue ):
                   ((i16, i32), ((i64, (i32, Struct,             )), Struct                 ), u16))
{
    zzz();
}

fn managed_box(&aa: &(int, int)) {
    zzz();
}

fn borrowed_pointer(&bb: &(int, int)) {
    zzz();
}

fn contained_borrowed_pointer((&cc, _): (&int, int)) {
    zzz();
}

fn unique_pointer(box dd: Box<(int, int, int)>) {
    zzz();
}

fn ref_binding(ref ee: (int, int, int)) {
    zzz();
}

fn ref_binding_in_tuple((ref ff, gg): (int, (int, int))) {
    zzz();
}

fn ref_binding_in_struct(Struct { b: ref hh, .. }: Struct) {
    zzz();
}

fn univariant_enum(Unit(ii): Univariant) {
    zzz();
}

fn univariant_enum_with_ref_binding(Unit(ref jj): Univariant) {
    zzz();
}

fn tuple_struct(TupleStruct(kk, ll): TupleStruct) {
    zzz();
}

fn tuple_struct_with_ref_binding(TupleStruct(mm, ref nn): TupleStruct) {
    zzz();
}

fn multiple_arguments((oo, pp): (int, int), qq : int) {
    zzz();
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
    unique_pointer(box() (40, 41, 42));
    ref_binding((43, 44, 45));
    ref_binding_in_tuple((46, (47, 48)));
    ref_binding_in_struct(Struct { a: 49, b: 50 });
    univariant_enum(Unit(51));
    univariant_enum_with_ref_binding(Unit(52));
    tuple_struct(TupleStruct(53.0, 54));
    tuple_struct_with_ref_binding(TupleStruct(55.0, 56));
    multiple_arguments((57, 58), 59);

    fn nested_function(rr: int, (ss, tt): (int, int)) {
        zzz();
    }

    nested_function(60, (61, 62));
}


fn zzz() {()}
