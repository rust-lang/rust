pub fn s() -> String {
    format!("s_upper_core -> ({})", m::m())
}

pub fn s_addr() -> usize { s as fn() -> String as *const u8 as usize }

pub fn m_addr() -> usize { m::m_addr() }

pub fn m_i_addr() -> usize { m::i_addr() }

pub fn m_j_addr() -> usize { m::j_addr() }

pub fn m_i_a_addr() -> usize { m::i_a_addr() }

pub fn m_j_a_addr() -> usize { m::j_a_addr() }
