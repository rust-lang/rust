use crate::ffi::OsString;
use crate::sys::pal::xous::os::get_application_parameters;
use crate::sys::pal::xous::os::params::ArgumentList;
use crate::{fmt, vec};

pub struct Args {
    parsed_args_list: vec::IntoIter<OsString>,
}

pub fn args() -> Args {
    let Some(params) = get_application_parameters() else {
        return Args { parsed_args_list: vec![].into_iter() };
    };

    for param in params {
        if let Ok(args) = ArgumentList::try_from(&param) {
            let mut parsed_args = vec![];
            for arg in args {
                parsed_args.push(arg.into());
            }
            return Args { parsed_args_list: parsed_args.into_iter() };
        }
    }
    Args { parsed_args_list: vec![].into_iter() }
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.parsed_args_list.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.parsed_args_list.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parsed_args_list.size_hint()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.parsed_args_list.next_back()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.parsed_args_list.len()
    }
}
