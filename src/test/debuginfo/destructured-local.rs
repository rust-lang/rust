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
// gdbg-check:$7 = {__0 = 6, __1 = 7}
// gdbr-check:$7 = (6, 7)

// gdb-command:print h
// gdb-check:$8 = 8
// gdb-command:print i
// gdbg-check:$9 = {a = 9, b = 10}
// gdbr-check:$9 = destructured_local::Struct {a: 9, b: 10}
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
// gdbg-check:$18 = {a = 21, b = 22}
// gdbr-check:$18 = destructured_local::Struct {a: 21, b: 22}

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
// gdbg-check:$30 = {__0 = 34, __1 = 35}
// gdbr-check:$30 = (34, 35)

// gdb-command:print bb
// gdbg-check:$31 = {__0 = 36, __1 = 37}
// gdbr-check:$31 = (36, 37)

// gdb-command:print cc
// gdb-check:$32 = 38

// gdb-command:print dd
// gdbg-check:$33 = {__0 = 40, __1 = 41, __2 = 42}
// gdbr-check:$33 = (40, 41, 42)

// gdb-command:print *ee
// gdbg-check:$34 = {__0 = 43, __1 = 44, __2 = 45}
// gdbr-check:$34 = (43, 44, 45)

// gdb-command:print *ff
// gdb-check:$35 = 46

// gdb-command:print gg
// gdbg-check:$36 = {__0 = 47, __1 = 48}
// gdbr-check:$36 = (47, 48)

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
// lldbg-check:[...]$0 = 1
// lldbr-check:(isize) a = 1
// lldb-command:print b
// lldbg-check:[...]$1 = false
// lldbr-check:(bool) b = false

// lldb-command:print c
// lldbg-check:[...]$2 = 2
// lldbr-check:(isize) c = 2
// lldb-command:print d
// lldbg-check:[...]$3 = 3
// lldbr-check:(u16) d = 3
// lldb-command:print e
// lldbg-check:[...]$4 = 4
// lldbr-check:(u16) e = 4

// lldb-command:print f
// lldbg-check:[...]$5 = 5
// lldbr-check:(isize) f = 5
// lldb-command:print g
// lldbg-check:[...]$6 = (6, 7)
// lldbr-check:((u32, u32)) g = { = 6 = 7 }

// lldb-command:print h
// lldbg-check:[...]$7 = 8
// lldbr-check:(i16) h = 8
// lldb-command:print i
// lldbg-check:[...]$8 = Struct { a: 9, b: 10 }
// lldbr-check:(destructured_local::Struct) i = Struct { a: 9, b: 10 }
// lldb-command:print j
// lldbg-check:[...]$9 = 11
// lldbr-check:(i16) j = 11

// lldb-command:print k
// lldbg-check:[...]$10 = 12
// lldbr-check:(i64) k = 12
// lldb-command:print l
// lldbg-check:[...]$11 = 13
// lldbr-check:(i32) l = 13

// lldb-command:print m
// lldbg-check:[...]$12 = 14
// lldbr-check:(i32) m = 14
// lldb-command:print n
// lldbg-check:[...]$13 = 16
// lldbr-check:(i32) n = 16

// lldb-command:print o
// lldbg-check:[...]$14 = 18
// lldbr-check:(i32) o = 18

// lldb-command:print p
// lldbg-check:[...]$15 = 19
// lldbr-check:(i64) p = 19
// lldb-command:print q
// lldbg-check:[...]$16 = 20
// lldbr-check:(i32) q = 20
// lldb-command:print r
// lldbg-check:[...]$17 = Struct { a: 21, b: 22 }
// lldbr-check:(destructured_local::Struct) r = Struct { a: 21, b: 22 }

// lldb-command:print s
// lldbg-check:[...]$18 = 24
// lldbr-check:(i32) s = 24
// lldb-command:print t
// lldbg-check:[...]$19 = 23
// lldbr-check:(i64) t = 23

// lldb-command:print u
// lldbg-check:[...]$20 = 25
// lldbr-check:(i32) u = 25
// lldb-command:print v
// lldbg-check:[...]$21 = 26
// lldbr-check:(i32) v = 26
// lldb-command:print w
// lldbg-check:[...]$22 = 27
// lldbr-check:(i32) w = 27
// lldb-command:print x
// lldbg-check:[...]$23 = 28
// lldbr-check:(i32) x = 28
// lldb-command:print y
// lldbg-check:[...]$24 = 29
// lldbr-check:(i64) y = 29
// lldb-command:print z
// lldbg-check:[...]$25 = 30
// lldbr-check:(i32) z = 30
// lldb-command:print ae
// lldbg-check:[...]$26 = 31
// lldbr-check:(i64) ae = 31
// lldb-command:print oe
// lldbg-check:[...]$27 = 32
// lldbr-check:(i32) oe = 32
// lldb-command:print ue
// lldbg-check:[...]$28 = 33
// lldbr-check:(i32) ue = 33

// lldb-command:print aa
// lldbg-check:[...]$29 = (34, 35)
// lldbr-check:((i32, i32)) aa = { = 34 = 35 }

// lldb-command:print bb
// lldbg-check:[...]$30 = (36, 37)
// lldbr-check:((i32, i32)) bb = { = 36 = 37 }

// lldb-command:print cc
// lldbg-check:[...]$31 = 38
// lldbr-check:(i32) cc = 38

// lldb-command:print dd
// lldbg-check:[...]$32 = (40, 41, 42)
// lldbr-check:((i32, i32, i32)) dd = { = 40 = 41 = 42 }

// lldb-command:print *ee
// lldbg-check:[...]$33 = (43, 44, 45)
// lldbr-check:((i32, i32, i32)) *ee = { = 43 = 44 = 45 }

// lldb-command:print *ff
// lldbg-check:[...]$34 = 46
// lldbr-check:(i32) *ff = 46

// lldb-command:print gg
// lldbg-check:[...]$35 = (47, 48)
// lldbr-check:((i32, i32)) gg = { = 47 = 48 }

// lldb-command:print *hh
// lldbg-check:[...]$36 = 50
// lldbr-check:(i32) *hh = 50

// lldb-command:print ii
// lldbg-check:[...]$37 = 51
// lldbr-check:(i32) ii = 51

// lldb-command:print *jj
// lldbg-check:[...]$38 = 52
// lldbr-check:(i32) *jj = 52

// lldb-command:print kk
// lldbg-check:[...]$39 = 53
// lldbr-check:(f64) kk = 53

// lldb-command:print ll
// lldbg-check:[...]$40 = 54
// lldbr-check:(isize) ll = 54

// lldb-command:print mm
// lldbg-check:[...]$41 = 55
// lldbr-check:(f64) mm = 55

// lldb-command:print *nn
// lldbg-check:[...]$42 = 56
// lldbr-check:(isize) *nn = 56


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
