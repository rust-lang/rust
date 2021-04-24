use crate::io;
use crate::sys;
use crate::sys::hermit::abi;

#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stdin {
    fn is_atty() -> bool {
        abi::isatty(abi::STDIN_FILENO)
    }
}

#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stdout {
    fn is_atty() -> bool {
        abi::isatty(abi::STDOUT_FILENO)
    }
}
#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stderr {
    fn is_atty() -> bool {
        abi::isatty(abi::STDERR_FILENO)
    }
}
