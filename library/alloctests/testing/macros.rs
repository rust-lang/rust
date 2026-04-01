macro_rules! struct_with_counted_drop {
    ($struct_name:ident $(( $( $elt_ty:ty ),+ ))?, $drop_counter:ident $( => $drop_stmt:expr )? ) => {
        thread_local! {static $drop_counter: ::core::cell::Cell<u32> = ::core::cell::Cell::new(0);}

        #[derive(Clone, Debug, PartialEq)]
        struct $struct_name $(( $( $elt_ty ),+ ))?;

        impl ::std::ops::Drop for $struct_name {
            fn drop(&mut self) {
                $drop_counter.set($drop_counter.get() + 1);

                $($drop_stmt(self))?
            }
        }
    };
    ($struct_name:ident $(( $( $elt_ty:ty ),+ ))?, $drop_counter:ident[ $drop_key:expr,$key_ty:ty ] $( => $drop_stmt:expr )? ) => {
        thread_local! {
            static $drop_counter: ::core::cell::RefCell<::std::collections::HashMap<$key_ty, u32>> =
                ::core::cell::RefCell::new(::std::collections::HashMap::new());
        }

        #[derive(Clone, Debug, PartialEq)]
        struct $struct_name $(( $( $elt_ty ),+ ))?;

        impl ::std::ops::Drop for $struct_name {
            fn drop(&mut self) {
                $drop_counter.with_borrow_mut(|counter| {
                    *counter.entry($drop_key(self)).or_default() += 1;
                });

                $($drop_stmt(self))?
            }
        }
    };
}

pub(crate) use struct_with_counted_drop;
