// check-fail

// issue #95698
macro_rules! from_cow_impls {
    ($( $from: ty ),+ $(,)? ) => {
        // recursion call
        from_cow_impls!(
            $( $from, Cow::from ),+
        );
        //~^^^ ERROR expansion grow limit reached while expanding `from_cow_impls!`
    };

    ($( $from: ty, $normalizer: expr ),+ $(,)? ) => {
        $( impl<'a> From<$from> for LhsValue<'a> {
            fn from(val: $from) -> Self {
                LhsValue::Bytes($normalizer(val))
            }
        } )+
    };
}

from_cow_impls!(
    &'a [u8], /*callback,*/
    Vec<u8>, /*callback,*/
);
