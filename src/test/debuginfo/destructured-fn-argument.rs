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
// lldbg-check:[...]$0 = 1
// lldbr-check:(isize) a = 1
// lldb-command:print b
// lldbg-check:[...]$1 = false
// lldbr-check:(bool) b = false
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$2 = 2
// lldbr-check:(isize) a = 2
// lldb-command:print b
// lldbg-check:[...]$3 = 3
// lldbr-check:(u16) b = 3
// lldb-command:print c
// lldbg-check:[...]$4 = 4
// lldbr-check:(u16) c = 4
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$5 = 5
// lldbr-check:(isize) a = 5
// lldb-command:print b
// lldbg-check:[...]$6 = { 0 = 6 1 = 7 }
// lldbr-check:((u32, u32)) b = { 0 = 6 1 = 7 }
// lldb-command:continue

// lldb-command:print h
// lldbg-check:[...]$7 = 8
// lldbr-check:(i16) h = 8
// lldb-command:print i
// lldbg-check:[...]$8 = { a = 9 b = 10 }
// lldbr-check:(destructured_fn_argument::Struct) i = { a = 9 b = 10 }
// lldb-command:print j
// lldbg-check:[...]$9 = 11
// lldbr-check:(i16) j = 11
// lldb-command:continue

// lldb-command:print k
// lldbg-check:[...]$10 = 12
// lldbr-check:(i64) k = 12
// lldb-command:print l
// lldbg-check:[...]$11 = 13
// lldbr-check:(i32) l = 13
// lldb-command:continue

// lldb-command:print m
// lldbg-check:[...]$12 = 14
// lldbr-check:(isize) m = 14
// lldb-command:print n
// lldbg-check:[...]$13 = 16
// lldbr-check:(i32) n = 16
// lldb-command:continue

// lldb-command:print o
// lldbg-check:[...]$14 = 18
// lldbr-check:(i32) o = 18
// lldb-command:continue

// lldb-command:print p
// lldbg-check:[...]$15 = 19
// lldbr-check:(i64) p = 19
// lldb-command:print q
// lldbg-check:[...]$16 = 20
// lldbr-check:(i32) q = 20
// lldb-command:print r
// lldbg-check:[...]$17 = { a = 21 b = 22 }
// lldbr-check:(destructured_fn_argument::Struct) r = { a = 21, b = 22 }
// lldb-command:continue

// lldb-command:print s
// lldbg-check:[...]$18 = 24
// lldbr-check:(i32) s = 24
// lldb-command:print t
// lldbg-check:[...]$19 = 23
// lldbr-check:(i64) t = 23
// lldb-command:continue

// lldb-command:print u
// lldbg-check:[...]$20 = 25
// lldbr-check:(i16) u = 25
// lldb-command:print v
// lldbg-check:[...]$21 = 26
// lldbr-check:(i32) v = 26
// lldb-command:print w
// lldbg-check:[...]$22 = 27
// lldbr-check:(i64) w = 27
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
// lldbr-check:(u16) ue = 33
// lldb-command:continue

// lldb-command:print aa
// lldbg-check:[...]$29 = { 0 = 34 1 = 35 }
// lldbr-check:((isize, isize)) aa = { 0 = 34 1 = 35 }
// lldb-command:continue

// lldb-command:print bb
// lldbg-check:[...]$30 = { 0 = 36 1 = 37 }
// lldbr-check:((isize, isize)) bb = { 0 = 36 1 = 37 }
// lldb-command:continue

// lldb-command:print cc
// lldbg-check:[...]$31 = 38
// lldbr-check:(isize) cc = 38
// lldb-command:continue

// lldb-command:print dd
// lldbg-check:[...]$32 = { 0 = 40 1 = 41 2 = 42 }
// lldbr-check:((isize, isize, isize)) dd = { 0 = 40 1 = 41 2 = 42 }
// lldb-command:continue

// lldb-command:print *ee
// lldbg-check:[...]$33 = { 0 = 43 1 = 44 2 = 45 }
// lldbr-check:((isize, isize, isize)) *ee = { 0 = 43 1 = 44 2 = 45 }
// lldb-command:continue

// lldb-command:print *ff
// lldbg-check:[...]$34 = 46
// lldbr-check:(isize) *ff = 46
// lldb-command:print gg
// lldbg-check:[...]$35 = { 0 = 47 1 = 48 }
// lldbr-check:((isize, isize)) gg = { 0 = 47 1 = 48 }
// lldb-command:continue

// lldb-command:print *hh
// lldbg-check:[...]$36 = 50
// lldbr-check:(i32) *hh = 50
// lldb-command:continue

// lldb-command:print ii
// lldbg-check:[...]$37 = 51
// lldbr-check:(i32) ii = 51
// lldb-command:continue

// lldb-command:print *jj
// lldbg-check:[...]$38 = 52
// lldbr-check:(i32) *jj = 52
// lldb-command:continue

// lldb-command:print kk
// lldbg-check:[...]$39 = 53
// lldbr-check:(f64) kk = 53
// lldb-command:print ll
// lldbg-check:[...]$40 = 54
// lldbr-check:(isize) ll = 54
// lldb-command:continue

// lldb-command:print mm
// lldbg-check:[...]$41 = 55
// lldbr-check:(f64) mm = 55
// lldb-command:print *nn
// lldbg-check:[...]$42 = 56
// lldbr-check:(isize) *nn = 56
// lldb-command:continue

// lldb-command:print oo
// lldbg-check:[...]$43 = 57
// lldbr-check:(isize) oo = 57
// lldb-command:print pp
// lldbg-check:[...]$44 = 58
// lldbr-check:(isize) pp = 58
// lldb-command:print qq
// lldbg-check:[...]$45 = 59
// lldbr-check:(isize) qq = 59
// lldb-command:continue

// lldb-command:print rr
// lldbg-check:[...]$46 = 60
// lldbr-check:(isize) rr = 60
// lldb-command:print ss
// lldbg-check:[...]$47 = 61
// lldbr-check:(isize) ss = 61
// lldb-command:print tt
// lldbg-check:[...]$48 = 62
// lldbr-check:(isize) tt = 62
// lldb-command:continue

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
    unique_pointer(Box::new((40, 41, 42)));
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
