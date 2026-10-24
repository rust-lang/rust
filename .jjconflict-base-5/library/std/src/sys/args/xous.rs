pub use super::common::Args;
use crate::sys::pal::params::{self, ArgumentList};

pub fn args() -> Args {
    let Some(params) = params::get() else {
        return Args::new(vec![]);
    };

    for param in params {
        if let Ok(args) = ArgumentList::try_from(&param) {
            let mut parsed_args = vec![];
            for arg in args {
                parsed_args.push(arg.into());
            }
            return Args::new(parsed_args);
        }
    }
    Args::new(vec![])
}
