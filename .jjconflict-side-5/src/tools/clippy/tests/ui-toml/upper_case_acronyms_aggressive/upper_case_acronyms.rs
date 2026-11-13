#![warn(clippy::upper_case_acronyms)]

struct HTTPResponse; // not linted by default, but with cfg option
//~^ upper_case_acronyms

struct CString; // not linted

enum Flags {
    NS, // not linted
    //~^ upper_case_acronyms
    CWR,
    //~^ upper_case_acronyms
    ECE,
    //~^ upper_case_acronyms
    URG,
    //~^ upper_case_acronyms
    ACK,
    //~^ upper_case_acronyms
    PSH,
    //~^ upper_case_acronyms
    RST,
    //~^ upper_case_acronyms
    SYN,
    //~^ upper_case_acronyms
    FIN,
    //~^ upper_case_acronyms
}

// linted with cfg option, beware that lint suggests `GccllvmSomething` instead of
// `GccLlvmSomething`
struct GCCLLVMSomething;
//~^ upper_case_acronyms

// don't warn on public items
pub struct MIXEDCapital;

pub struct FULLCAPITAL;

// enum variants should not be linted if the num is pub
pub enum ParseError<T> {
    FULLCAPITAL(u8),
    MIXEDCapital(String),
    Utf8(std::string::FromUtf8Error),
    Parse(T, String),
}

// private, do lint here
enum ParseErrorPrivate<T> {
    WASD(u8),
    //~^ upper_case_acronyms
    WASDMixed(String),
    //~^ upper_case_acronyms
    Utf8(std::string::FromUtf8Error),
    Parse(T, String),
}

fn main() {}
