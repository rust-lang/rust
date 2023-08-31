#![warn(clippy::upper_case_acronyms)]

struct HTTPResponse; // not linted by default, but with cfg option

struct CString; // not linted

enum Flags {
    NS, // not linted
    CWR,
    //~^ ERROR: name `CWR` contains a capitalized acronym
    //~| NOTE: `-D clippy::upper-case-acronyms` implied by `-D warnings`
    ECE,
    //~^ ERROR: name `ECE` contains a capitalized acronym
    URG,
    //~^ ERROR: name `URG` contains a capitalized acronym
    ACK,
    //~^ ERROR: name `ACK` contains a capitalized acronym
    PSH,
    //~^ ERROR: name `PSH` contains a capitalized acronym
    RST,
    //~^ ERROR: name `RST` contains a capitalized acronym
    SYN,
    //~^ ERROR: name `SYN` contains a capitalized acronym
    FIN,
    //~^ ERROR: name `FIN` contains a capitalized acronym
}

// linted with cfg option, beware that lint suggests `GccllvmSomething` instead of
// `GccLlvmSomething`
struct GCCLLVMSomething;

// public items must not be linted
pub struct NOWARNINGHERE;
pub struct ALSONoWarningHERE;

// enum variants should not be linted if the num is pub
pub enum ParseError<T> {
    YDB(u8),
    Utf8(std::string::FromUtf8Error),
    Parse(T, String),
}

// private, do lint here
enum ParseErrorPrivate<T> {
    WASD(u8),
    //~^ ERROR: name `WASD` contains a capitalized acronym
    Utf8(std::string::FromUtf8Error),
    Parse(T, String),
}

// do lint here
struct JSON;
//~^ ERROR: name `JSON` contains a capitalized acronym

// do lint here
enum YAML {
    //~^ ERROR: name `YAML` contains a capitalized acronym
    Num(u32),
    Str(String),
}

fn main() {}
