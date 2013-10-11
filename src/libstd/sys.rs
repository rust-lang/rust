// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Misc low level stuff

#[allow(missing_doc)];

use any::Any;
use kinds::Send;
use rt::task::{UnwindReasonStr, UnwindReasonAny};
use rt::task;
use send_str::{SendStr, IntoSendStr};

/// Trait for initiating task failure with a sendable cause.
pub trait FailWithCause {
    /// Fail the current task with `cause`.
    fn fail_with(cause: Self, file: &'static str, line: uint) -> !;
}

impl FailWithCause for ~str {
    fn fail_with(cause: ~str, file: &'static str, line: uint) -> ! {
        task::begin_unwind_reason(UnwindReasonStr(cause.into_send_str()), file, line)
    }
}

impl FailWithCause for &'static str {
    fn fail_with(cause: &'static str, file: &'static str, line: uint) -> ! {
        task::begin_unwind_reason(UnwindReasonStr(cause.into_send_str()), file, line)
    }
}

impl FailWithCause for SendStr {
    fn fail_with(cause: SendStr, file: &'static str, line: uint) -> ! {
        task::begin_unwind_reason(UnwindReasonStr(cause), file, line)
    }
}

impl FailWithCause for ~Any {
    fn fail_with(cause: ~Any, file: &'static str, line: uint) -> ! {
        task::begin_unwind_reason(UnwindReasonAny(cause), file, line)
    }
}

impl<T: Any + Send + 'static> FailWithCause for ~T {
    fn fail_with(cause: ~T, file: &'static str, line: uint) -> ! {
        task::begin_unwind_reason(UnwindReasonAny(cause as ~Any), file, line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use any::Any;
    use cast;
    use send_str::IntoSendStr;

    #[test]
    fn synthesize_closure() {
        use unstable::raw::Closure;
        unsafe {
            let x = 10;
            let f: &fn(int) -> int = |y| x + y;

            assert_eq!(f(20), 30);

            let original_closure: Closure = cast::transmute(f);

            let actual_function_pointer = original_closure.code;
            let environment = original_closure.env;

            let new_closure = Closure {
                code: actual_function_pointer,
                env: environment
            };

            let new_f: &fn(int) -> int = cast::transmute(new_closure);
            assert_eq!(new_f(20), 30);
        }
    }

    #[test]
    #[should_fail]
    fn fail_static() { FailWithCause::fail_with("cause", file!(), line!()) }

    #[test]
    #[should_fail]
    fn fail_owned() { FailWithCause::fail_with(~"cause", file!(), line!()) }

    #[test]
    #[should_fail]
    fn fail_send() { FailWithCause::fail_with("cause".into_send_str(), file!(), line!()) }

    #[test]
    #[should_fail]
    fn fail_any() { FailWithCause::fail_with(~612_u16 as ~Any, file!(), line!()) }

    #[test]
    #[should_fail]
    fn fail_any_wrap() { FailWithCause::fail_with(~413_u16, file!(), line!()) }
}
