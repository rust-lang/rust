#![allow(dead_code)]
#![deny(improper_ctypes)]

use std::marker::PhantomData;

#[repr(C)]
pub struct ZeroSize;
#[repr(transparent)]
pub struct ZeroSizeTransparent;
#[repr(C)]
pub struct ZeroSizeWithPhantomData(::std::marker::PhantomData<i32>);
#[repr(transparent)]
pub struct TransparentPhantomData(::std::marker::PhantomData<i32>);
pub struct RustyPhantomData(::std::marker::PhantomData<i32>);

#[repr(C)]
pub struct HasPhantomData<U>(f32, PhantomData<U>);


#[repr(transparent)]
pub struct TransparentUnit<U>(f32, PhantomData<U>);
#[repr(transparent)]
pub struct TransparentUnit2S<U>((), PhantomData<U>);
#[repr(transparent)]
pub struct TransparentUnit2<T,U>(T, PhantomData<U>);
#[repr(transparent)]
pub struct TransparentUnit3(f32, ());
#[repr(C)]
pub struct CReprUnit<U>(f32, PhantomData<U>);
#[repr(C)]
pub struct CReprUnit2<U>((), PhantomData<U>);
#[repr(C)]
pub struct CReprUnit3(f32, ());

#[repr(C)]
pub struct WrapperForPhantoms<T>(T,u32);

#[repr(C,packed(2))]
pub struct BadPackedStruct(u8,u16,u32);
#[repr(C,align(8))]
pub struct BadAlignedStruct(u8,u16,u32);

pub union RUnion {
    a: (u8,u8),
    b: u16,
}
#[repr(C)]
pub union CUnion {
    a: [u8;2],
    b: u16,
}
#[repr(C,packed(2))]  // FIXME: does this repr even make sense from a memory layout point of view?
pub union PUnion {
    a: (u8,u8),
    b: u16,
}
#[repr(C,align(8))]
pub union AUnion {
    a: (u8,u8),
    b: u16,
}

extern "C" {
    pub fn transp1(p: TransparentUnit<ZeroSize>);
    // note: 2 and 2s are the same type (technically) but type params messes with 1-ZST detection
    pub fn transp2(p: TransparentUnit2<(), u8>); //~ ERROR uses type `TransparentUnit2<(), u8>`
    pub fn transp2s(p: TransparentUnit2S<u8>); //~ ERROR uses type `TransparentUnit2S<u8>`
    pub fn transp2b(p: TransparentUnit2<ZeroSize, ()>);
    //~^ ERROR uses type `TransparentUnit2<ZeroSize, ()>`
    //~| ERROR uses type `TransparentUnit2<ZeroSize, ()>`
    pub fn transp4(p: TransparentUnit3);
    pub fn crepr1(p: CReprUnit<ZeroSize>);
    pub fn crepr2(p: CReprUnit2<ZeroSize>);
    pub fn crepr3(p: CReprUnit3);

    pub fn phantom(p: HasPhantomData<f64>);

    pub fn zero_size(p: ZeroSize); //~ ERROR uses type `ZeroSize`
    pub fn zero_size_transp(p: ZeroSizeTransparent); //~ ERROR uses type `ZeroSizeTransparent`

    pub fn zero_size_phantom(p: ZeroSizeWithPhantomData);
    //~^ ERROR uses type `ZeroSizeWithPhantomData`
    pub fn transparent_phantom(p: TransparentPhantomData);
    //~^ ERROR uses type `TransparentPhantomData`
    pub fn rusty_phantom(p: RustyPhantomData);
    //~^ ERROR uses type `RustyPhantomData`

    //TODO: those three results are probably wrong... (check sizes of all-phantom structs)
    pub fn zero_size_phantom_wrapped(p: WrapperForPhantoms<ZeroSizeWithPhantomData>);
    pub fn transparent_phantom_wrapped(p: WrapperForPhantoms<TransparentPhantomData>);
    pub fn rusty_phantom_wrapped(p: WrapperForPhantoms<RustyPhantomData>);
    //~^ ERROR uses type `WrapperForPhantoms<RustyPhantomData>`

    //TODO: fix the bad alignment shtuff
    pub fn bad_packed(p: BadPackedStruct); //~ ERROR uses type `BadPackedStruct`
    pub fn bad_aligned(p: BadAlignedStruct); //~ ERROR uses type `BadAlignedStruct`

    pub fn unions(p1: RUnion, p2: CUnion, p3: PUnion, p4: AUnion);
    //~^ ERROR: uses type `RUnion`
    //~| ERROR: uses type `PUnion`
    //~| ERROR: uses type `AUnion`
}

fn main(){}
