pub fn z() -> String {
    format!("z_roof_core -> ({}), -> ({})", s::s(), t::t())
}

pub fn z_addr() -> usize { z as fn() -> String as *const u8 as usize }

pub fn s_addr() -> usize { s::s_addr() }

pub fn t_addr() -> usize { t::t_addr() }

pub fn s_m_addr() -> usize { s::m_addr() }

pub fn t_m_addr() -> usize { t::m_addr() }

pub fn s_m_i_addr() -> usize { s::m_i_addr() }

pub fn s_m_j_addr() -> usize { s::m_j_addr() }

pub fn t_m_i_addr() -> usize { t::m_i_addr() }

pub fn t_m_j_addr() -> usize { t::m_j_addr() }

pub fn s_m_i_a_addr() -> usize { s::m_i_a_addr() }

pub fn s_m_j_a_addr() -> usize { s::m_j_a_addr() }

pub fn t_m_i_a_addr() -> usize { t::m_i_a_addr() }

pub fn t_m_j_a_addr() -> usize { t::m_j_a_addr() }
