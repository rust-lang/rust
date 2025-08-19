//@ compile-flags:-g
//@ disable-gdb-pretty-printers

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
// gdb-check:$7 = (6, 7)

// gdb-command:print h
// gdb-check:$8 = 8
// gdb-command:print i
// gdb-check:$9 = destructured_local::Struct {a: 9, b: 10}
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
// gdb-check:$18 = destructured_local::Struct {a: 21, b: 22}

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
// gdb-check:$30 = (34, 35)

// gdb-command:print bb
// gdb-check:$31 = (36, 37)

// gdb-command:print cc
// gdb-check:$32 = 38

// gdb-command:print dd
// gdb-check:$33 = (40, 41, 42)

// gdb-command:print *ee
// gdb-check:$34 = (43, 44, 45)

// gdb-command:print *ff
// gdb-check:$35 = 46

// gdb-command:print gg
// gdb-check:$36 = (47, 48)

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
// lldb-check:[...] 1
// lldb-command:v b
// lldb-check:[...] false

// lldb-command:v c
// lldb-check:[...] 2
// lldb-command:v d
// lldb-check:[...] 3
// lldb-command:v e
// lldb-check:[...] 4

// lldb-command:v f
// lldb-check:[...] 5
// lldb-command:v g
// lldb-check:[...] { 0 = 6 1 = 7 }

// lldb-command:v h
// lldb-check:[...] 8
// lldb-command:v i
// lldb-check:[...] { a = 9 b = 10 }
// lldb-command:v j
// lldb-check:[...] 11

// lldb-command:v k
// lldb-check:[...] 12
// lldb-command:v l
// lldb-check:[...] 13

// lldb-command:v m
// lldb-check:[...] 14
// lldb-command:v n
// lldb-check:[...] 16

// lldb-command:v o
// lldb-check:[...] 18

// lldb-command:v p
// lldb-check:[...] 19
// lldb-command:v q
// lldb-check:[...] 20
// lldb-command:v r
// lldb-check:[...] { a = 21 b = 22 }

// lldb-command:v s
// lldb-check:[...] 24
// lldb-command:v t
// lldb-check:[...] 23

// lldb-command:v u
// lldb-check:[...] 25
// lldb-command:v v
// lldb-check:[...] 26
// lldb-command:v w
// lldb-check:[...] 27
// lldb-command:v x
// lldb-check:[...] 28
// lldb-command:v y
// lldb-check:[...] 29
// lldb-command:v z
// lldb-check:[...] 30
// lldb-command:v ae
// lldb-check:[...] 31
// lldb-command:v oe
// lldb-check:[...] 32
// lldb-command:v ue
// lldb-check:[...] 33

// lldb-command:v aa
// lldb-check:[...] { 0 = 34 1 = 35 }

// lldb-command:v bb
// lldb-check:[...] { 0 = 36 1 = 37 }

// lldb-command:v cc
// lldb-check:[...] 38

// lldb-command:v dd
// lldb-check:[...] { 0 = 40 1 = 41 2 = 42 }

// lldb-command:v *ee
// lldb-check:[...] { 0 = 43 1 = 44 2 = 45 }

// lldb-command:v *ff
// lldb-check:[...] 46

// lldb-command:v gg
// lldb-check:[...] { 0 = 47 1 = 48 }

// lldb-command:v *hh
// lldb-check:[...] 50

// lldb-command:v ii
// lldb-check:[...] 51

// lldb-command:v *jj
// lldb-check:[...] 52

// lldb-command:v kk
// lldb-check:[...] 53

// lldb-command:v ll
// lldb-check:[...] 54

// lldb-command:v mm
// lldb-check:[...] 55

// lldb-command:v *nn
// lldb-check:[...] 56


#![allow(unused_variables)]
#![feature(box_patterns)]

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
