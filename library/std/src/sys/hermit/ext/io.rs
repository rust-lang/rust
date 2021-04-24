use crate::io;
use crate::sys;
use crate::sys::hermit::abi;

#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stdin {
    fn is_terminal() -> bool {
        abi::isatty(abi::STDIN_FILENO)
    }
}

#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stdout {
    fn is_terminal() -> bool {
        abi::isatty(abi::STDOUT_FILENO)
    }
}
#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stderr {
    fn is_terminal() -> bool {
        abi::isatty(abi::STDERR_FILENO)
    }
}
