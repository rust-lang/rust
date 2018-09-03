/// CONTROL register
pub struct CONTROL;

rsr!(CONTROL);
wsr!(CONTROL);

/// Execution Program Status Register
pub struct EPSR;

rsr!(EPSR);

/// Interrupt Program Status Register
pub struct IPSR;

rsr!(IPSR);

/// Main Stack Pointer
pub struct MSP;

rsrp!(MSP);
wsrp!(MSP);

/// Priority Mask Register
pub struct PRIMASK;

rsr!(PRIMASK);
wsr!(PRIMASK);

/// Process Stack Pointer
pub struct PSP;

rsrp!(PSP);
wsrp!(PSP);

/// Program Status Register
#[allow(non_camel_case_types)]
pub struct xPSR;

rsr!(xPSR);
