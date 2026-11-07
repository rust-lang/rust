// ignore-tidy-linelength
//@ normalize-stderr: "values of the type `[^`]+` are too big" -> "values of the type $$REALLY_TOO_BIG are too big"


#![feature(transmute_generic_consts)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const W: usize, const H: usize>(v: [[u32; H + 1]; W]) -> [[u32; W + 1]; H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR cannot transmute
    }
}

fn bar<const W: bool, const H: usize>(v: [[u32; H]; W]) -> [[u32; W]; H] {
    //~^ ERROR: the constant `W` is not of type `usize`
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: the constant `W` is not of type `usize`
    }
}

fn baz<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [u32; W * H * H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR cannot transmute
    }
}

fn overflow(v: [[[u32; 8888888]; 9999999]; 777777777]) -> [[[u32; 9999999]; 777777777]; 8888888] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR cannot transmute
    }
}

fn overflow_more(v: [[[u32; 8888888]; 9999999]; 777777777]) -> [[[u32; 9999999]; 777777777]; 239] {
    unsafe { std::mem::transmute(v) } //~ ERROR cannot transmute between types of different sizes
}


fn transpose<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [[u32; W]; H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn ident<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [[u32; H]; W] {
    unsafe { std::mem::transmute(v) }
}

fn flatten<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [u32; W * H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn coagulate<const W: usize, const H: usize>(v: [u32; H * W]) -> [[u32; W]; H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn flatten_3d<const W: usize, const H: usize, const D: usize>(
    v: [[[u32; D]; H]; W],
) -> [u32; D * W * H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn flatten_somewhat<const W: usize, const H: usize, const D: usize>(
    v: [[[u32; D]; H]; W],
) -> [[u32; D * W]; H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn known_size<const L: usize>(v: [u16; L]) -> [u8; L * 2] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn condense_bytes<const L: usize>(v: [u8; L * 2]) -> [u16; L] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn singleton_each<const L: usize>(v: [u8; L]) -> [[u8; 1]; L] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn transpose_with_const<const W: usize, const H: usize>(
    v: [[u32; 2 * H]; W + W],
) -> [[u32; W + W]; 2 * H] {
    unsafe {
        std::mem::transmute(v)
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
    }
}

fn main() {}
