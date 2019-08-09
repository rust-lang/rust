// run-pass
// This test deserializes an enum in-place by transmuting to a union that
// should have the same layout, and manipulating the tag and payloads
// independently. This verifies that `repr(some_int)` has a stable representation,
// and that we don't miscompile these kinds of manipulations.

use std::time::Duration;
use std::mem;

#[repr(C, u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum MyEnum {
    A(u32),                     // Single primitive value
    B { x: u8, y: i16, z: u8 }, // Composite, and the offsets of `y` and `z`
                                // depend on tag being internal
    C,                          // Empty
    D(Option<u32>),             // Contains an enum
    E(Duration),                // Contains a struct
}

#[repr(C)]
struct MyEnumRepr {
    tag: MyEnumTag,
    payload: MyEnumPayload,
}

#[repr(C)]
#[allow(non_snake_case)]
union MyEnumPayload {
    A: MyEnumVariantA,
    B: MyEnumVariantB,
    D: MyEnumVariantD,
    E: MyEnumVariantE,
}

#[repr(u8)] #[derive(Copy, Clone)] enum MyEnumTag { A, B, C, D, E }
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantA(u32);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantB {x: u8, y: i16, z: u8 }
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantD(Option<u32>);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantE(Duration);

fn main() {
    let result: Vec<Result<MyEnum, ()>> = vec![
        Ok(MyEnum::A(17)),
        Ok(MyEnum::B { x: 206, y: 1145, z: 78 }),
        Ok(MyEnum::C),
        Err(()),
        Ok(MyEnum::D(Some(407))),
        Ok(MyEnum::D(None)),
        Ok(MyEnum::E(Duration::from_secs(100))),
        Err(()),
    ];

    // Binary serialized version of the above (little-endian)
    let input: Vec<u8> = vec![
        0,  17, 0, 0, 0,
        1,  206,  121, 4,  78,
        2,
        8,  /* invalid tag value */
        3,  0,  151, 1, 0, 0,
        3,  1,
        4,  100, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
        0,  /* incomplete value */
    ];

    let mut output = vec![];
    let mut buf = &input[..];

    unsafe {
        // This should be safe, because we don't match on it unless it's fully formed,
        // and it doesn't have a destructor.
        #[allow(deprecated)]
        let mut dest: MyEnum = mem::uninitialized();
        while buf.len() > 0 {
            match parse_my_enum(&mut dest, &mut buf) {
                Ok(()) => output.push(Ok(dest)),
                Err(()) => output.push(Err(())),
            }
        }
    }

    assert_eq!(output, result);
}

fn parse_my_enum<'a>(dest: &'a mut MyEnum, buf: &mut &[u8]) -> Result<(), ()> {
    unsafe {
        // Should be correct to do this transmute.
        let dest: &'a mut MyEnumRepr = mem::transmute(dest);
        let tag = read_u8(buf)?;

        dest.tag = match tag {
            0 => MyEnumTag::A,
            1 => MyEnumTag::B,
            2 => MyEnumTag::C,
            3 => MyEnumTag::D,
            4 => MyEnumTag::E,
            _ => return Err(()),
        };

        match dest.tag {
            MyEnumTag::A => {
                dest.payload.A.0 = read_u32_le(buf)?;
            }
            MyEnumTag::B => {
                dest.payload.B.x = read_u8(buf)?;
                dest.payload.B.y = read_u16_le(buf)? as i16;
                dest.payload.B.z = read_u8(buf)?;
            }
            MyEnumTag::C => {
                /* do nothing */
            }
            MyEnumTag::D => {
                let is_some = read_u8(buf)? == 0;
                if is_some {
                    dest.payload.D.0 = Some(read_u32_le(buf)?);
                } else {
                    dest.payload.D.0 = None;
                }
            }
            MyEnumTag::E => {
                let secs = read_u64_le(buf)?;
                let nanos = read_u32_le(buf)?;
                dest.payload.E.0 = Duration::new(secs, nanos);
            }
        }
        Ok(())
    }
}



// reader helpers

fn read_u64_le(buf: &mut &[u8]) -> Result<u64, ()> {
    if buf.len() < 8 { return Err(()) }
    let val = (buf[0] as u64) << 0
            | (buf[1] as u64) << 8
            | (buf[2] as u64) << 16
            | (buf[3] as u64) << 24
            | (buf[4] as u64) << 32
            | (buf[5] as u64) << 40
            | (buf[6] as u64) << 48
            | (buf[7] as u64) << 56;
    *buf = &buf[8..];
    Ok(val)
}

fn read_u32_le(buf: &mut &[u8]) -> Result<u32, ()> {
    if buf.len() < 4 { return Err(()) }
    let val = (buf[0] as u32) << 0
            | (buf[1] as u32) << 8
            | (buf[2] as u32) << 16
            | (buf[3] as u32) << 24;
    *buf = &buf[4..];
    Ok(val)
}

fn read_u16_le(buf: &mut &[u8]) -> Result<u16, ()> {
    if buf.len() < 2 { return Err(()) }
    let val = (buf[0] as u16) << 0
            | (buf[1] as u16) << 8;
    *buf = &buf[2..];
    Ok(val)
}

fn read_u8(buf: &mut &[u8]) -> Result<u8, ()> {
    if buf.len() < 1 { return Err(()) }
    let val = buf[0];
    *buf = &buf[1..];
    Ok(val)
}
