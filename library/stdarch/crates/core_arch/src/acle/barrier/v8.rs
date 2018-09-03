/// Full system is the required shareability domain, reads are the required
/// access type
pub struct LD;

dmb_dsb!(LD);

/// Inner Shareable is the required shareability domain, reads are the required
/// access type
pub struct ISHLD;

dmb_dsb!(ISHLD);

/// Non-shareable is the required shareability domain, reads are the required
/// access type
pub struct NSHLD;

dmb_dsb!(NSHLD);

/// Outher Shareable is the required shareability domain, reads are the required
/// access type
pub struct OSHLD;

dmb_dsb!(OSHLD);
