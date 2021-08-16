pub fn i() -> String {
    format!("i_ground_core -> ({})", a::a())
}

pub fn i_addr() -> usize { i as fn() -> String as *const u8 as usize }

pub fn a_addr() -> usize { a::a_addr() }
