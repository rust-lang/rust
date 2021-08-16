pub fn m() -> String {
    format!("m_middle_core -> ({}), -> ({})", i::i(), j::j())
}

pub fn m_addr() -> usize { m as fn() -> String as *const u8 as usize }

pub fn i_addr() -> usize { i::i_addr() }

pub fn j_addr() -> usize { j::j_addr() }

pub fn i_a_addr() -> usize { i::a_addr() }

pub fn j_a_addr() -> usize { j::a_addr() }
