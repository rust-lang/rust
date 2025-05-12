#![warn(clippy::upper_case_acronyms)]

struct HTTPResponse; // not linted by default, but with cfg option

struct CString; // not linted

enum Flags {
    NS, // not linted
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
    //~^ upper_case_acronyms
    Utf8(std::string::FromUtf8Error),
    Parse(T, String),
}

// do lint here
struct JSON;
//~^ upper_case_acronyms

// do lint here
enum YAML {
    //~^ upper_case_acronyms
    Num(u32),
    Str(String),
}

// test for issue #7708
enum AllowOnField {
    DISALLOW,
    //~^ upper_case_acronyms
    #[allow(clippy::upper_case_acronyms)]
    ALLOW,
}

fn main() {}
