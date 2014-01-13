// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10381)

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print a
// check:$1 = 1
// debugger:print b
// check:$2 = false

// debugger:print c
// check:$3 = 2
// debugger:print d
// check:$4 = 3
// debugger:print e
// check:$5 = 4

// debugger:print f
// check:$6 = 5
// debugger:print g
// check:$7 = {6, 7}

// debugger:print h
// check:$8 = 8
// debugger:print i
// check:$9 = {a = 9, b = 10}
// debugger:print j
// check:$10 = 11

// debugger:print k
// check:$11 = 12
// debugger:print l
// check:$12 = 13

// debugger:print m
// check:$13 = 14
// debugger:print n
// check:$14 = 16

// debugger:print o
// check:$15 = 18

// debugger:print p
// check:$16 = 19
// debugger:print q
// check:$17 = 20
// debugger:print r
// check:$18 = {a = 21, b = 22}

// debugger:print s
// check:$19 = 24
// debugger:print t
// check:$20 = 23

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

// debugger:print aa
// check:$30 = {34, 35}

// debugger:print bb
// check:$31 = {36, 37}

// debugger:print cc
// check:$32 = 38

// debugger:print dd
// check:$33 = {40, 41, 42}

// debugger:print *ee
// check:$34 = {43, 44, 45}

// debugger:print *ff
// check:$35 = 46

// debugger:print gg
// check:$36 = {47, 48}

// debugger:print *hh
// check:$37 = 50

// debugger:print ii
// check:$38 = 51

// debugger:print *jj
// check:$39 = 52

// debugger:print kk
// check:$40 = 53

// debugger:print ll
// check:$41 = 54

// debugger:print mm
// check:$42 = 55

// debugger:print *nn
// check:$43 = 56

#[allow(unused_variable)];

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
    let (m, _, n) = (14, 15, 16);

    // ignored struct field
    let Struct { b: o, .. } = Struct { a: 17, b: 18 };

    // one struct destructured, one not
    let (Struct { a: p, b: q }, r) = (Struct { a: 19, b: 20 }, Struct { a: 21, b: 22 });

    // different order of struct fields
    let Struct { b: s, a: t } = Struct { a: 23, b: 24 };

    // complex nesting
    let ((u, v), ((w, (x, Struct { a: y, b: z})), Struct { a: ae, b: oe }), ue) =
        ((25, 26), ((27, (28, Struct { a: 29, b: 30})), Struct { a: 31, b: 32 }), 33);

    // reference
    let &aa = &(34, 35);

    // reference
    let &bb = &(36, 37);

    // contained reference
    let (&cc, _) = (&38, 39);

    // unique pointer
    let ~dd = ~(40, 41, 42);

    // ref binding
    let ref ee = (43, 44, 45);

    // ref binding in tuple
    let (ref ff, gg) = (46, (47, 48));

    // ref binding in struct
    let Struct { b: ref hh, .. } = Struct { a: 49, b: 50 };

    // univariant enum
    let Unit(ii) = Unit(51);

    // univariant enum with ref      binding
    let Unit(ref jj) = Unit(52);

    // tuple struct
    let TupleStruct(kk, ll) = TupleStruct(53.0, 54);

    // tuple struct with ref binding
    let TupleStruct(mm, ref nn) = TupleStruct(55.0, 56);

    zzz();
}

fn zzz() {()}
