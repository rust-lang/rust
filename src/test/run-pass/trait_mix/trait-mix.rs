import u_trait_mix;

impl f32: u_trait_mix::num {
    pure fn add(&&other: f32)    -> f32 { return self + other; }
    pure fn sub(&&other: f32)    -> f32 { return self - other; }
    pure fn mul(&&other: f32)    -> f32 { return self * other; }
    pure fn div(&&other: f32)    -> f32 { return self / other; }
    pure fn modulo(&&other: f32) -> f32 { return self % other; }
    pure fn neg()                -> f32 { return -self;        }

    pure fn to_int()         -> int { return self as int; }
    static pure fn from_int(n: int) -> f32 { return n as f32;    }
}

/*
It seems that this will fail if I try using it from another crate.

*/

/*

// ICEs if I put this in num -- ???
trait from_int {

}
*/

fn main() {}
