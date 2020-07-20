// Public traits for CommandSized. May x.py have mercy on me.
use crate::ffi::OsStr;
use crate::io;

#[cfg(any(windows, doc))]
use crate::ffi::OsString;

/// Traits for handling a sized command.
#[unstable(feature = "command_sized", issue = "74549")]
pub trait CommandSized: core::marker::Sized {
    /// Possibly pass an argument.
    /// Returns an error if the size of the arguments would overflow the command line. The error contains the reason the remaining arguments could not be added.
    fn maybe_arg(&mut self, arg: impl Arg) -> io::Result<&mut Self>;
    /// Possibly pass many arguments.
    /// Returns an error if the size of the arguments would overflow the command line. The error contains the number of arguments added as well as the reason the remaining arguments could not be added.
    fn maybe_args(
        &mut self,
        args: &mut impl Iterator<Item = impl Arg>,
    ) -> Result<&mut Self, (usize, io::Error)>;
    /// Build multiple commands to consume all arguments.
    /// Returns an error if the size of an argument would overflow the command line. The error contains the reason the remaining arguments could not be added.
    fn xargs<I, S, A>(
        program: S,
        args: &mut I,
        before: Vec<A>,
        after: Vec<A>,
    ) -> io::Result<Vec<Self>>
    where
        I: Iterator<Item = A>,
        S: AsRef<OsStr> + Copy,
        A: Arg;
}

/// Types that can be appended to a Windows command-line. Used for custom escaping.
///
/// Do not implement this trait on Unix. Its existence is only due to the way CommandSized
/// is defined.
// FIXME: The force-quoted one should conceptually be its own type. Would be
// iseful for xargs.
#[unstable(feature = "command_sized", issue = "74549")]
pub trait Arg {
    /// Calculate size of arg in a cmdline.
    fn arg_size(&self, force_quotes: bool) -> Result<usize, Problem>;
    #[cfg(any(unix))]
    #[doc(cfg(unix))]
    /// Convert to more palatable form for Unix.
    fn to_plain(&self) -> &OsStr;
    #[cfg(any(windows))]
    #[doc(cfg(windows))]
    /// Retain the argument by copying. Wait, why we are retaining it?
    // FIXME: Isn't information already lost when we put it into the
    // vector, erasing type info? Why do we still use the args vector?
    // Okay, apparently we are putting it in a CommandArgs<>. Hmm.
    fn to_os_string(&self) -> OsString;
    #[cfg(any(windows))]
    #[doc(cfg(windows))]
    /// Append argument to a cmdline.
    fn append_to(&self, cmd: &mut Vec<u16>, force_quotes: bool) -> Result<usize, Problem>;
}

#[derive(Copy, Clone)]
#[unstable(feature = "command_sized", issue = "74549")]
pub enum Problem {
    SawNul,
    Oversized,
}

#[unstable(feature = "command_sized", issue = "74549")]
impl From<&Problem> for io::Error {
    fn from(problem: &Problem) -> io::Error {
        match *problem {
            Problem::SawNul => {
                io::Error::new(io::ErrorKind::InvalidInput, "nul byte found in provided data")
            }
            Problem::Oversized => {
                io::Error::new(io::ErrorKind::InvalidInput, "command exceeds maximum size")
            }
        }
    }
}

#[unstable(feature = "command_sized", issue = "74549")]
impl From<Problem> for io::Error {
    fn from(problem: Problem) -> io::Error {
        (&problem).into()
    }
}

/// Implementation for the above trait.
macro_rules! impl_command_sized {
    (prelude) => {
        use crate::sys::process::{Arg, Problem};
        use core::convert::TryFrom;
    };
    (marg $marg_func:path) => {
        fn maybe_arg(&mut self, arg: impl Arg) -> io::Result<&mut Self> {
            $marg_func(self.as_inner_mut(), arg)?;
            Ok(self)
        }
    };
    (margs $marg_func:path) => {
        fn maybe_args(
            &mut self,
            args: &mut impl Iterator<Item = impl Arg>,
        ) -> Result<&mut Self, (usize, io::Error)> {
            let mut count: usize = 0;
            for arg in args {
                if let Err(err) = $marg_func(self.as_inner_mut(), arg) {
                    return Err((count, err));
                }
                count += 1;
            }
            Ok(self)
        }
    };
    (xargs $args_func:path) => {
        fn xargs<I, S, A>(
            program: S,
            args: &mut I,
            before: Vec<A>,
            after: Vec<A>,
        ) -> io::Result<Vec<Self>>
        where
            I: Iterator<Item = A>,
            S: AsRef<OsStr> + Copy,
            A: Arg,
        {
            let mut ret = Vec::new();
            let mut cmd = Self::new(program);
            let mut fresh: bool = true;

            // This performs a nul check.
            let tail_size: usize = after
                .iter()
                .map(|x| Arg::arg_size(x, false))
                .collect::<Result<Vec<_>, Problem>>()?
                .iter()
                .sum();

            if let Err(_) = isize::try_from(tail_size) {
                return Err(Problem::Oversized.into());
            }

            $args_func(&mut cmd, &before);
            if cmd.as_inner_mut().available_size(false)? < (tail_size as isize) {
                return Err(Problem::Oversized.into());
            }

            for arg in args {
                let size = arg.arg_size(false)?;
                // Negative case is catched outside of loop.
                if (cmd.as_inner_mut().available_size(false)? as usize) < (size + tail_size) {
                    if fresh {
                        return Err(Problem::Oversized.into());
                    }
                    $args_func(&mut cmd, &after);
                    ret.push(cmd);
                    cmd = Self::new(program);
                    $args_func(&mut cmd, &before);
                }
                cmd.maybe_arg(arg)?;
                fresh = false;
            }
            $args_func(&mut cmd, &after);
            ret.push(cmd);
            Ok(ret)
        }
    };
}
