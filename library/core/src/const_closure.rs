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
pub(crate) struct ConstFnMutClosure<CapturedData, Function> {
    /// The Data captured by the Closure.
    /// Must be either a (mutable) reference or a tuple of (mutable) references.
    pub data: CapturedData,
    /// The Function of the Closure, must be: Fn(CapturedData, ClosureArgs) -> ClosureReturn
    pub func: Function,
}
impl<'a, CapturedData: ?Sized, Function> ConstFnMutClosure<&'a mut CapturedData, Function> {
    /// Function for creating a new closure.
    ///
    /// `data` is the a mutable borrow of data that is captured from the environment.
    /// If you want Data to be a tuple of mutable Borrows, the struct must be constructed manually.
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

macro_rules! impl_fn_mut_tuple {
    ($($var:ident)*) => {
        #[allow(unused_parens)]
        impl<'a, $($var,)* ClosureArguments, Function, ClosureReturnValue> const
            FnOnce<ClosureArguments> for ConstFnMutClosure<($(&'a mut $var),*), Function>
        where
            Function: ~const Fn(($(&mut $var),*), ClosureArguments) -> ClosureReturnValue+ ~const Destruct,
        {
            type Output = ClosureReturnValue;

            extern "rust-call" fn call_once(mut self, args: ClosureArguments) -> Self::Output {
            self.call_mut(args)
            }
        }
        #[allow(unused_parens)]
        impl<'a, $($var,)* ClosureArguments, Function, ClosureReturnValue> const
            FnMut<ClosureArguments> for ConstFnMutClosure<($(&'a mut $var),*), Function>
        where
            Function: ~const Fn(($(&mut $var),*), ClosureArguments)-> ClosureReturnValue,
        {
            extern "rust-call" fn call_mut(&mut self, args: ClosureArguments) -> Self::Output {
                #[allow(non_snake_case)]
                let ($($var),*) = &mut self.data;
                (self.func)(($($var),*), args)
            }
        }
    };
}
impl_fn_mut_tuple!(A);
impl_fn_mut_tuple!(A B);
impl_fn_mut_tuple!(A B C);
impl_fn_mut_tuple!(A B C D);
impl_fn_mut_tuple!(A B C D E);
