/// Base Priority Mask Register
pub struct BASEPRI;

rsr!(BASEPRI);
wsr!(BASEPRI);

/// Base Priority Mask Register (conditional write)
#[allow(non_camel_case_types)]
pub struct BASEPRI_MAX;

wsr!(BASEPRI_MAX);

/// Fault Mask Register
pub struct FAULTMASK;

rsr!(FAULTMASK);
wsr!(FAULTMASK);
