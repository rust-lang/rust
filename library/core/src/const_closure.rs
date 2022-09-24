use crate::marker::Destruct;

/// Struct representing a closure with mutably borrowed data.
///
/// Example:
/// ```no_build
/// #![feature(const_mut_refs)]
/// use crate::const_closure::ConstFnMutClosure;
/// const fn imp(state: &mut i32, (arg,): (i32,)) -> i32 {
///   *state += arg;
///   *state
/// }
/// let mut i = 5;
/// let mut cl = ConstFnMutClosure::new(&mut i, imp);
///
/// assert!(7 == cl(2));
/// assert!(8 == cl(1));
/// ```
pub(crate) struct ConstFnMutClosure<'a, CapturedData: ?Sized, Function> {
    data: &'a mut CapturedData,
    func: Function,
}

impl<'a, CapturedData: ?Sized, Function> ConstFnMutClosure<'a, CapturedData, Function> {
    /// Function for creating a new closure.
    ///
    /// `data` is the a mutable borrow of data that is captured from the environment.
    ///
    /// `func` is the function of the closure, it gets the data and a tuple of the arguments closure
    ///   and return the return value of the closure.
    pub(crate) const fn new<ClosureArguments, ClosureReturnValue>(
        data: &'a mut CapturedData,
        func: Function,
    ) -> Self
    where
        Function: ~const Fn(&mut CapturedData, ClosureArguments) -> ClosureReturnValue,
    {
        Self { data, func }
    }
}

impl<'a, CapturedData: ?Sized, ClosureArguments, Function, ClosureReturnValue> const
    FnOnce<ClosureArguments> for ConstFnMutClosure<'a, CapturedData, Function>
where
    Function:
        ~const Fn(&mut CapturedData, ClosureArguments) -> ClosureReturnValue + ~const Destruct,
{
    type Output = ClosureReturnValue;

    extern "rust-call" fn call_once(mut self, args: ClosureArguments) -> Self::Output {
        self.call_mut(args)
    }
}

impl<'a, CapturedData: ?Sized, ClosureArguments, Function, ClosureReturnValue> const
    FnMut<ClosureArguments> for ConstFnMutClosure<'a, CapturedData, Function>
where
    Function: ~const Fn(&mut CapturedData, ClosureArguments) -> ClosureReturnValue,
{
    extern "rust-call" fn call_mut(&mut self, args: ClosureArguments) -> Self::Output {
        (self.func)(self.data, args)
    }
}
