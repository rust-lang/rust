#![feature(repr_simd)]
#![feature(no_niche)]

#[repr(transparent())] //~ ERROR invalid `repr(transparent)` attribute: no arguments expected
                       //~| ERROR invalid `repr(transparent)` attribute: no arguments expected
struct S0(u32);

#[repr(transparent(1, 2, 3))] //~ ERROR invalid `repr(transparent)` attribute: no arguments expected
                              //~| ERROR invalid `repr(transparent)` attribute: no arguments expected
struct S1(u32);

#[repr(C())] //~ ERROR invalid `repr(C)` attribute: no arguments expected
             //~| ERROR invalid `repr(C)` attribute: no arguments expected
struct S2(u32);

#[repr(C(1, 2, 3))] //~ ERROR invalid `repr(C)` attribute: no arguments expected
                    //~| ERROR invalid `repr(C)` attribute: no arguments expected
struct S3(u32);

#[repr(simd())] //~ ERROR invalid `repr(simd)` attribute: no arguments expected
                //~| ERROR invalid `repr(simd)` attribute: no arguments expected
struct S4(u32);

#[repr(simd(1, 2, 3))] //~ ERROR invalid `repr(simd)` attribute: no arguments expected
                       //~| ERROR invalid `repr(simd)` attribute: no arguments expected
struct S5(u32);

#[repr(no_niche())] //~ ERROR invalid `repr(no_niche)` attribute: no arguments expected
                    //~| ERROR invalid `repr(no_niche)` attribute: no arguments expected
struct S6(u32);

#[repr(no_niche(1, 2, 3))] //~ ERROR invalid `repr(no_niche)` attribute: no arguments expected
                           //~| ERROR invalid `repr(no_niche)` attribute: no arguments expected
struct S7(u32);


#[repr(align)] //~ ERROR invalid `repr(align)` attribute: expected a non-negative number
               //~| ERROR invalid `repr(align)` attribute: expected a non-negative number
struct S8(u32);

#[repr(align())] //~ ERROR invalid `repr(align)` attribute: expected a non-negative number
                 //~| ERROR invalid `repr(align)` attribute: expected a non-negative number
struct S9(u32);

#[repr(align(foo()))] //~ ERROR invalid `repr(align)` attribute
                      //~| ERROR invalid `repr(align)` attribute
struct S10(u32);

#[repr(align = "1")] //~ ERROR incorrect `repr(align)` attribute
                     //~| ERROR incorrect `repr(align)` attribute
struct S11(u32);

#[repr(align = "")] //~ ERROR incorrect `repr(align)` attribute
                    //~| ERROR incorrect `repr(align)` attribute
struct S12(u32);

#[repr(align = true)] //~ ERROR incorrect `repr(align)` attribute
                      //~| ERROR incorrect `repr(align)` attribute
struct S13(u32);

#[repr(align(1, 2, 3))] //~ ERROR invalid `repr(align)` attribute: expected only one value
                        //~| ERROR invalid `repr(align)` attribute: expected only one value
struct S14(u32);

#[repr(packed())] //~ ERROR invalid `repr(packed)` attribute: expected a non-negative number
                  //~| ERROR invalid `repr(packed)` attribute: expected a non-negative number
struct S15(u32);

#[repr(packed(1, 2, 3))] //~ ERROR invalid `repr(packed)` attribute: expected only one value
                         //~| ERROR invalid `repr(packed)` attribute: expected only one value
struct S16(u32);

#[repr(i8())] //~ ERROR invalid `repr(i8)` attribute: no arguments expected
              //~| ERROR invalid `repr(i8)` attribute: no arguments expected
enum S17 { A, B }

#[repr(i8(1, 2, 3))] //~ ERROR invalid `repr(i8)` attribute: no arguments expected
                     //~| ERROR invalid `repr(i8)` attribute: no arguments expected
enum S18 { A, B }

#[repr] //~ ERROR malformed `repr` attribute input
struct S19(u32);

#[repr(123)] //~ ERROR meta item in `repr` must be an identifier
struct S20(u32);

#[repr("foo")] //~ ERROR meta item in `repr` must be an identifier
struct S21(u32);

fn main() {}
