fn static_to_a_to_static_through_ref_in_tuple<'a>(x: &'a u32) -> &'static u32 {
    let (ref y, _z): (&'a u32, u32) = (&22, 44);
    *y //~ ERROR
}

fn main() {}
