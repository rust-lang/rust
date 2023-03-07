// run-pass

use std::mem;

// Univariant C-like enum
#[repr(i32)]
enum Univariant {
    X = 17
}

#[repr(u16)]
enum UnivariantWithoutDescr {
    Y
}

#[repr(u8)]
enum UnivariantWithData {
    Z(u8),
}

pub fn main() {
    {
        assert_eq!(4, mem::size_of::<Univariant>());
        assert_eq!(17, Univariant::X as i32);

        let enums: &[Univariant] =
            &[Univariant::X, Univariant::X, Univariant::X];
        let ints: &[i32] = unsafe { mem::transmute(enums) };
        // check it has the same memory layout as i32
        assert_eq!(&[17, 17, 17], ints);
    }

    {
        assert_eq!(2, mem::size_of::<UnivariantWithoutDescr>());
        let descr = UnivariantWithoutDescr::Y as u16;

        let enums: &[UnivariantWithoutDescr] =
            &[UnivariantWithoutDescr::Y, UnivariantWithoutDescr::Y, UnivariantWithoutDescr::Y];
        let ints: &[u16] = unsafe { mem::transmute(enums) };
        // check it has the same memory layout as u16
        assert_eq!(&[descr, descr, descr], ints);
    }

    {
        assert_eq!(2, mem::size_of::<UnivariantWithData>());

        match UnivariantWithData::Z(4) {
            UnivariantWithData::Z(x) => assert_eq!(x, 4),
        }
    }
}
