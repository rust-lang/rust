//@ min-lldb-version: 310

//@ compile-flags:-g

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

// lldb-command:v a
// lldbg-check:[...] 1
// lldbr-check:(isize) a = 1
// lldb-command:v b
// lldbg-check:[...] false
// lldbr-check:(bool) b = false

// lldb-command:v c
// lldbg-check:[...] 2
// lldbr-check:(isize) c = 2
// lldb-command:v d
// lldbg-check:[...] 3
// lldbr-check:(u16) d = 3
// lldb-command:v e
// lldbg-check:[...] 4
// lldbr-check:(u16) e = 4

// lldb-command:v f
// lldbg-check:[...] 5
// lldbr-check:(isize) f = 5
// lldb-command:v g
// lldbg-check:[...] { 0 = 6 1 = 7 }
// lldbr-check:((u32, u32)) g = { 0 = 6 1 = 7 }

// lldb-command:v h
// lldbg-check:[...] 8
// lldbr-check:(i16) h = 8
// lldb-command:v i
// lldbg-check:[...] { a = 9 b = 10 }
// lldbr-check:(destructured_local::Struct) i = { a = 9 b = 10 }
// lldb-command:v j
// lldbg-check:[...] 11
// lldbr-check:(i16) j = 11

// lldb-command:v k
// lldbg-check:[...] 12
// lldbr-check:(i64) k = 12
// lldb-command:v l
// lldbg-check:[...] 13
// lldbr-check:(i32) l = 13

// lldb-command:v m
// lldbg-check:[...] 14
// lldbr-check:(i32) m = 14
// lldb-command:v n
// lldbg-check:[...] 16
// lldbr-check:(i32) n = 16

// lldb-command:v o
// lldbg-check:[...] 18
// lldbr-check:(i32) o = 18

// lldb-command:v p
// lldbg-check:[...] 19
// lldbr-check:(i64) p = 19
// lldb-command:v q
// lldbg-check:[...] 20
// lldbr-check:(i32) q = 20
// lldb-command:v r
// lldbg-check:[...] { a = 21 b = 22 }
// lldbr-check:(destructured_local::Struct) r = { a = 21 b = 22 }

// lldb-command:v s
// lldbg-check:[...] 24
// lldbr-check:(i32) s = 24
// lldb-command:v t
// lldbg-check:[...] 23
// lldbr-check:(i64) t = 23

// lldb-command:v u
// lldbg-check:[...] 25
// lldbr-check:(i32) u = 25
// lldb-command:v v
// lldbg-check:[...] 26
// lldbr-check:(i32) v = 26
// lldb-command:v w
// lldbg-check:[...] 27
// lldbr-check:(i32) w = 27
// lldb-command:v x
// lldbg-check:[...] 28
// lldbr-check:(i32) x = 28
// lldb-command:v y
// lldbg-check:[...] 29
// lldbr-check:(i64) y = 29
// lldb-command:v z
// lldbg-check:[...] 30
// lldbr-check:(i32) z = 30
// lldb-command:v ae
// lldbg-check:[...] 31
// lldbr-check:(i64) ae = 31
// lldb-command:v oe
// lldbg-check:[...] 32
// lldbr-check:(i32) oe = 32
// lldb-command:v ue
// lldbg-check:[...] 33
// lldbr-check:(i32) ue = 33

// lldb-command:v aa
// lldbg-check:[...] { 0 = 34 1 = 35 }
// lldbr-check:((i32, i32)) aa = { 0 = 34 1 = 35 }

// lldb-command:v bb
// lldbg-check:[...] { 0 = 36 1 = 37 }
// lldbr-check:((i32, i32)) bb = { 0 = 36 1 = 37 }

// lldb-command:v cc
// lldbg-check:[...] 38
// lldbr-check:(i32) cc = 38

// lldb-command:v dd
// lldbg-check:[...] { 0 = 40 1 = 41 2 = 42 }
// lldbr-check:((i32, i32, i32)) dd = { 0 = 40 1 = 41 2 = 42}

// lldb-command:v *ee
// lldbg-check:[...] { 0 = 43 1 = 44 2 = 45 }
// lldbr-check:((i32, i32, i32)) *ee = { 0 = 43 1 = 44 2 = 45}

// lldb-command:v *ff
// lldbg-check:[...] 46
// lldbr-check:(i32) *ff = 46

// lldb-command:v gg
// lldbg-check:[...] { 0 = 47 1 = 48 }
// lldbr-check:((i32, i32)) gg = { 0 = 47 1 = 48 }

// lldb-command:v *hh
// lldbg-check:[...] 50
// lldbr-check:(i32) *hh = 50

// lldb-command:v ii
// lldbg-check:[...] 51
// lldbr-check:(i32) ii = 51

// lldb-command:v *jj
// lldbg-check:[...] 52
// lldbr-check:(i32) *jj = 52

// lldb-command:v kk
// lldbg-check:[...] 53
// lldbr-check:(f64) kk = 53

// lldb-command:v ll
// lldbg-check:[...] 54
// lldbr-check:(isize) ll = 54

// lldb-command:v mm
// lldbg-check:[...] 55
// lldbr-check:(f64) mm = 55

// lldb-command:v *nn
// lldbg-check:[...] 56
// lldbr-check:(isize) *nn = 56


#![allow(unused_variables)]
#![feature(box_patterns)]
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
    let box dd = Box::new((40, 41, 42));

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
