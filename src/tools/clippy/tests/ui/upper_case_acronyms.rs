#![warn(clippy::upper_case_acronyms)]

struct HTTPResponse; // not linted by default, but with cfg option

struct CString; // not linted

enum Flags {
    NS, // not linted
    CWR,
    ECE,
    URG,
    ACK,
    PSH,
    RST,
    SYN,
    FIN,
}

// linted with cfg option, beware that lint suggests `GccllvmSomething` instead of
// `GccLlvmSomething`
struct GCCLLVMSomething;

// public items must not be linted
pub struct NOWARNINGHERE;
pub struct ALSONoWarningHERE;

fn main() {}
