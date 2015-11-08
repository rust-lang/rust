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

// gdb-command:print c
// gdb-check:$3 = 2
// gdb-command:print d
// gdb-check:$4 = 3
// gdb-command:print e
// gdb-check:$5 = 4

// gdb-command:print f
// gdb-check:$6 = 5
// gdb-command:print g
// gdb-check:$7 = {__0 = 6, __1 = 7}

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
// gdb-check:$30 = {__0 = 34, __1 = 35}

// gdb-command:print bb
// gdb-check:$31 = {__0 = 36, __1 = 37}

// gdb-command:print cc
// gdb-check:$32 = 38

// gdb-command:print dd
// gdb-check:$33 = {__0 = 40, __1 = 41, __2 = 42}

// gdb-command:print *ee
// gdb-check:$34 = {__0 = 43, __1 = 44, __2 = 45}

// gdb-command:print *ff
// gdb-check:$35 = 46

// gdb-command:print gg
// gdb-check:$36 = {__0 = 47, __1 = 48}

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


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print a
// lldb-check:[...]$0 = 1
// lldb-command:print b
// lldb-check:[...]$1 = false

// lldb-command:print c
// lldb-check:[...]$2 = 2
// lldb-command:print d
// lldb-check:[...]$3 = 3
// lldb-command:print e
// lldb-check:[...]$4 = 4

// lldb-command:print f
// lldb-check:[...]$5 = 5
// lldb-command:print g
// lldb-check:[...]$6 = (6, 7)

// lldb-command:print h
// lldb-check:[...]$7 = 8
// lldb-command:print i
// lldb-check:[...]$8 = Struct { a: 9, b: 10 }
// lldb-command:print j
// lldb-check:[...]$9 = 11

// lldb-command:print k
// lldb-check:[...]$10 = 12
// lldb-command:print l
// lldb-check:[...]$11 = 13

// lldb-command:print m
// lldb-check:[...]$12 = 14
// lldb-command:print n
// lldb-check:[...]$13 = 16

// lldb-command:print o
// lldb-check:[...]$14 = 18

// lldb-command:print p
// lldb-check:[...]$15 = 19
// lldb-command:print q
// lldb-check:[...]$16 = 20
// lldb-command:print r
// lldb-check:[...]$17 = Struct { a: 21, b: 22 }

// lldb-command:print s
// lldb-check:[...]$18 = 24
// lldb-command:print t
// lldb-check:[...]$19 = 23

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

// lldb-command:print aa
// lldb-check:[...]$29 = (34, 35)

// lldb-command:print bb
// lldb-check:[...]$30 = (36, 37)

// lldb-command:print cc
// lldb-check:[...]$31 = 38

// lldb-command:print dd
// lldb-check:[...]$32 = (40, 41, 42)

// lldb-command:print *ee
// lldb-check:[...]$33 = (43, 44, 45)

// lldb-command:print *ff
// lldb-check:[...]$34 = 46

// lldb-command:print gg
// lldb-check:[...]$35 = (47, 48)

// lldb-command:print *hh
// lldb-check:[...]$36 = 50

// lldb-command:print ii
// lldb-check:[...]$37 = 51

// lldb-command:print *jj
// lldb-check:[...]$38 = 52

// lldb-command:print kk
// lldb-check:[...]$39 = 53

// lldb-command:print ll
// lldb-check:[...]$40 = 54

// lldb-command:print mm
// lldb-check:[...]$41 = 55

// lldb-command:print *nn
// lldb-check:[...]$42 = 56


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


fn main() {
    // simple tuple
    let (a, b) : (isize, bool) = (1, false);

    // nested tuple
    let (c, (d, e)) : (isize, (u16, u16)) = (2, (3, 4));

    // bind tuple-typed value to one name (destructure only first level)
    let (f, g) : (isize, (u32, u32)) = (5, (6, 7));

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
    let box dd = box (40, 41, 42);

    // ref binding
    let ref ee = (43, 44, 45);

    // ref binding in tuple
    let (ref ff, gg) = (46, (47, 48));

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

    zzz(); // #break
}

fn zzz() { () }
