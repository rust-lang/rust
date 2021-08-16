pub fn j() -> String {
    format!("j_ground_core -> ({})", a::a())
}

pub fn j_addr() -> usize { j as fn() -> String as *const u8 as usize }

pub fn a_addr() -> usize { a::a_addr() }
