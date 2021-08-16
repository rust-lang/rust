// ignore-test Not a test. Used by the other tests.

pub fn sanity_check() {
    // Sanity-check: getting back non-trival values from addr functions
    assert_ne!(z::s_addr(), z::t_addr());
    assert_ne!(z::s_m_addr(), z::s_m_i_addr());
    assert_ne!(z::s_m_i_addr(), z::s_m_j_addr());
    assert_ne!(z::s_m_i_addr(), z::s_m_i_a_addr());
}

pub fn check_linked_function_equivalence() {
    // Check that the linked functions are the same code by comparing their
    // underlying addresses.
    assert_eq!(z::s_m_addr(), z::s::m_addr());
    assert_eq!(z::s_m_addr(), z::t_m_addr());
    assert_eq!(z::s_m_i_addr(), z::s::m::i_addr());
    assert_eq!(z::s_m_i_addr(), z::t_m_i_addr());
    assert_eq!(z::s_m_i_a_addr(), z::s::m::i::a_addr());
    assert_eq!(z::s_m_i_a_addr(), z::t_m_i_a_addr());
    assert_eq!(z::s_m_i_a_addr(), z::t_m_j_a_addr());
}
