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

// gdb-command:print c
// gdb-check:$3 = 2
// gdb-command:print d
// gdb-check:$4 = 3
// gdb-command:print e
// gdb-check:$5 = 4

// gdb-command:print f
// gdb-check:$6 = 5
// gdb-command:print g
// gdb-check:$7 = {6, 7}

// gdb-command:print h
// gdb-check:$8 = 8
// gdb-command:print i
// gdb-check:$9 = {a = 9, b = 10}
// gdb-command:print j
// gdb-check:$10 = 11

// gdb-command:print k
// gdb-check:$11 = 12
// gdb-command:print l
// gdb-check:$12 = 13

// gdb-command:print m
// gdb-check:$13 = 14
// gdb-command:print n
// gdb-check:$14 = 16

// gdb-command:print o
// gdb-check:$15 = 18

// gdb-command:print p
// gdb-check:$16 = 19
// gdb-command:print q
// gdb-check:$17 = 20
// gdb-command:print r
// gdb-check:$18 = {a = 21, b = 22}

// gdb-command:print s
// gdb-check:$19 = 24
// gdb-command:print t
// gdb-check:$20 = 23

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

// gdb-command:print aa
// gdb-check:$30 = {34, 35}

// gdb-command:print bb
// gdb-check:$31 = {36, 37}

// gdb-command:print cc
// gdb-check:$32 = 38

// gdb-command:print dd
// gdb-check:$33 = {40, 41, 42}

// gdb-command:print *ee
// gdb-check:$34 = {43, 44, 45}

// gdb-command:print *ff
// gdb-check:$35 = 46

// gdb-command:print gg
// gdb-check:$36 = {47, 48}

// gdb-command:print *hh
// gdb-check:$37 = 50

// gdb-command:print ii
// gdb-check:$38 = 51

// gdb-command:print *jj
// gdb-check:$39 = 52

// gdb-command:print kk
// gdb-check:$40 = 53

// gdb-command:print ll
// gdb-check:$41 = 54

// gdb-command:print mm
// gdb-check:$42 = 55

// gdb-command:print *nn
// gdb-check:$43 = 56

#![allow(unused_variable)]

struct Struct {
    a: i64,
    b: i32
}

enum Univariant {
    Unit(i32)
}

struct TupleStruct (f64, int);


fn main() {
    // simple tuple
    let (a, b) : (int, bool) = (1, false);

    // nested tuple
    let (c, (d, e)) : (int, (u16, u16)) = (2, (3, 4));

    // bind tuple-typed value to one name (destructure only first level)
    let (f, g) : (int, (u32, u32)) = (5, (6, 7));

    // struct as tuple element
    let (h, i, j) : (i16, Struct, i16) = (8, Struct { a: 9, b: 10 }, 11);

    // struct pattern
    let Struct { a: k, b: l } = Struct { a: 12, b: 13 };

    // ignored tuple element
    let (m, _, n) = (14i, 15i, 16i);

    // ignored struct field
    let Struct { b: o, .. } = Struct { a: 17, b: 18 };

    // one struct destructured, one not
    let (Struct { a: p, b: q }, r) = (Struct { a: 19, b: 20 }, Struct { a: 21, b: 22 });

    // different order of struct fields
    let Struct { b: s, a: t } = Struct { a: 23, b: 24 };

    // complex nesting
    let ((u, v), ((w, (x, Struct { a: y, b: z})), Struct { a: ae, b: oe }), ue) =
        ((25i, 26i), ((27i, (28i, Struct { a: 29, b: 30})), Struct { a: 31, b: 32 }), 33i);

    // reference
    let &aa = &(34i, 35i);

    // reference
    let &bb = &(36i, 37i);

    // contained reference
    let (&cc, _) = (&38i, 39i);

    // unique pointer
    let box dd = box() (40i, 41i, 42i);

    // ref binding
    let ref ee = (43i, 44i, 45i);

    // ref binding in tuple
    let (ref ff, gg) = (46i, (47i, 48i));

    // ref binding in struct
    let Struct { b: ref hh, .. } = Struct { a: 49, b: 50 };

    // univariant enum
    let Unit(ii) = Unit(51);

    // univariant enum with ref      binding
    let &Unit(ref jj) = &Unit(52);

    // tuple struct
    let &TupleStruct(kk, ll) = &TupleStruct(53.0, 54);

    // tuple struct with ref binding
    let &TupleStruct(mm, ref nn) = &TupleStruct(55.0, 56);

    zzz();
}

fn zzz() {()}
