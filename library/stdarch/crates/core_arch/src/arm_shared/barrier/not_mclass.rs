//! Access types available on v7 and v8 but not on v7(E)-M or v8-M

/// Full system is the required shareability domain, writes are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct ST;

dmb_dsb!(ST);

/// Inner Shareable is the required shareability domain, reads and writes are
/// the required access types
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct ISH;

dmb_dsb!(ISH);

/// Inner Shareable is the required shareability domain, writes are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct ISHST;

dmb_dsb!(ISHST);

/// Non-shareable is the required shareability domain, reads and writes are the
/// required access types
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct NSH;

dmb_dsb!(NSH);

/// Non-shareable is the required shareability domain, writes are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct NSHST;

dmb_dsb!(NSHST);

/// Outer Shareable is the required shareability domain, reads and writes are
/// the required access types
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct OSH;

dmb_dsb!(OSH);

/// Outer Shareable is the required shareability domain, writes are the required
/// access type
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct OSHST;

dmb_dsb!(OSHST);
