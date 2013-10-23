// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run

// debugger:finish
// debugger:print a
// check:$1 = 1
// debugger:print b
// check:$2 = false
// debugger:continue

// debugger:finish
// debugger:print a
// check:$3 = 2
// debugger:print b
// check:$4 = 3
// debugger:print c
// check:$5 = 4
// debugger:continue

// debugger:finish
// debugger:print a
// check:$6 = 5
// debugger:print b
// check:$7 = {6, 7}
// debugger:continue

// debugger:finish
// debugger:print h
// check:$8 = 8
// debugger:print i
// check:$9 = {a = 9, b = 10}
// debugger:print j
// check:$10 = 11
// debugger:continue

// debugger:finish
// debugger:print k
// check:$11 = 12
// debugger:print l
// check:$12 = 13
// debugger:continue

// debugger:finish
// debugger:print m
// check:$13 = 14
// debugger:print n
// check:$14 = 16
// debugger:continue

// debugger:finish
// debugger:print o
// check:$15 = 18
// debugger:continue

// debugger:finish
// debugger:print p
// check:$16 = 19
// debugger:print q
// check:$17 = 20
// debugger:print r
// check:$18 = {a = 21, b = 22}
// debugger:continue

// debugger:finish
// debugger:print s
// check:$19 = 24
// debugger:print t
// check:$20 = 23
// debugger:continue

// debugger:finish
// debugger:print u
// check:$21 = 25
// debugger:print v
// check:$22 = 26
// debugger:print w
// check:$23 = 27
// debugger:print x
// check:$24 = 28
// debugger:print y
// check:$25 = 29
// debugger:print z
// check:$26 = 30
// debugger:print ae
// check:$27 = 31
// debugger:print oe
// check:$28 = 32
// debugger:print ue
// check:$29 = 33
// debugger:continue

// debugger:finish
// debugger:print aa
// check:$30 = {34, 35}
// debugger:continue

// debugger:finish
// debugger:print bb
// check:$31 = {36, 37}
// debugger:continue

// debugger:finish
// debugger:print cc
// check:$32 = 38
// debugger:continue

// debugger:finish
// debugger:print dd
// check:$33 = {40, 41, 42}
// debugger:continue

// debugger:finish
// debugger:print *ee
// check:$34 = {43, 44, 45}
// debugger:continue

// debugger:finish
// debugger:print *ff
// check:$35 = 46
// debugger:print gg
// check:$36 = {47, 48}
// debugger:continue

// debugger:finish
// debugger:print *hh
// check:$37 = 50
// debugger:continue

// debugger:finish
// debugger:print ii
// check:$38 = 51
// debugger:continue

// debugger:finish
// debugger:print *jj
// check:$39 = 52
// debugger:continue

// debugger:finish
// debugger:print kk
// check:$40 = 53
// debugger:print ll
// check:$41 = 54
// debugger:continue

// debugger:finish
// debugger:print mm
// check:$42 = 55
// debugger:print *nn
// check:$43 = 56
// debugger:continue

// debugger:finish
// debugger:print oo
// check:$44 = 57
// debugger:print pp
// check:$45 = 58
// debugger:print qq
// check:$46 = 59
// debugger:continue

// debugger:finish
// debugger:print rr
// check:$47 = 60
// debugger:print ss
// check:$48 = 61
// debugger:print tt
// check:$49 = 62
// debugger:continue

#[allow(unused_variable)];

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

fn ignored_struct_field(Struct { b: o, _ }: Struct) {
    zzz();
}

fn one_struct_destructured_one_not((Struct { a: p, b: q }, r): (Struct, Struct)) {
    zzz();
}

fn different_order_of_struct_fields(Struct { b: s, a: t }: Struct ) {
    zzz();
}

fn complex_nesting(((u,   v  ), ((w,   (x,   Struct { a: y, b: z})), Struct { a: ae, b: oe }), ue ):
                   ((i16, i32), ((i64, (i32, Struct,             )), Struct                 ), u16)) {
    zzz();
}

fn managed_box(@aa: @(int, int)) {
    zzz();
}

fn borrowed_pointer(&bb: &(int, int)) {
    zzz();
}

fn contained_borrowed_pointer((&cc, _): (&int, int)) {
    zzz();
}

fn unique_pointer(~dd: ~(int, int, int)) {
    zzz();
}

fn ref_binding(ref ee: (int, int, int)) {
    zzz();
}

fn ref_binding_in_tuple((ref ff, gg): (int, (int, int))) {
    zzz();
}

fn ref_binding_in_struct(Struct { b: ref hh, _ }: Struct) {
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
    managed_box(@(34, 35));
    borrowed_pointer(&(36, 37));
    contained_borrowed_pointer((&38, 39));
    unique_pointer(~(40, 41, 42));
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
