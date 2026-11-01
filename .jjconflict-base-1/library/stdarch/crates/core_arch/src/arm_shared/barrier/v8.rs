/// Full system is the required shareability domain, reads are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct LD;

dmb_dsb!(LD);

/// Inner Shareable is the required shareability domain, reads are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct ISHLD;

dmb_dsb!(ISHLD);

/// Non-shareable is the required shareability domain, reads are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct NSHLD;

dmb_dsb!(NSHLD);

/// Outer Shareable is the required shareability domain, reads are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct OSHLD;

dmb_dsb!(OSHLD);
