// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

#![feature(generators, negative_impls, rustc_attrs)]

macro_rules! type_combinations {
    (
        $( $name:ident => { $( $tt:tt )* } );* $(;)?
    ) => { $(
        mod $name {
            $( $tt )*

            impl !Sync for Client {}
            impl !Send for Client {}
        }

        // Struct update syntax. This fails because the Client used in the update is considered
        // dropped *after* the yield.
        {
            let g = move || match drop($name::Client { ..$name::Client::default() }) {
            //~^ `significant_drop::Client` which is not `Send`
            //~| `insignificant_dtor::Client` which is not `Send`
            //[no_drop_tracking,drop_tracking]~| `derived_drop::Client` which is not `Send`
                _ => yield,
            };
            assert_send(g);
            //~^ ERROR cannot be sent between threads
            //~| ERROR cannot be sent between threads
            //~| ERROR cannot be sent between threads
            //[no_drop_tracking]~| ERROR cannot be sent between threads
        }

        // Simple owned value. This works because the Client is considered moved into `drop`,
        // even though the temporary expression doesn't end until after the yield.
        {
            let g = move || match drop($name::Client::default()) {
                _ => yield,
            };
            assert_send(g);
            //[no_drop_tracking]~^ ERROR cannot be sent between threads
            //[no_drop_tracking]~| ERROR cannot be sent between threads
            //[no_drop_tracking]~| ERROR cannot be sent between threads
            //[no_drop_tracking]~| ERROR cannot be sent between threads
        }
    )* }
}

fn assert_send<T: Send>(_thing: T) {}

fn main() {
    type_combinations!(
        // OK
        copy => { #[derive(Copy, Clone, Default)] pub struct Client; };
        // NOT OK: MIR borrowck thinks that this is used after the yield, even though
        // this has no `Drop` impl and only the drops of the fields are observable.
        // FIXME: this should compile.
        derived_drop => { #[derive(Default)] pub struct Client { pub nickname: String } };
        // NOT OK
        significant_drop => {
            #[derive(Default)]
            pub struct Client;
            impl Drop for Client {
                fn drop(&mut self) {}
            }
        };
        // NOT OK (we need to agree with MIR borrowck)
        insignificant_dtor => {
            #[derive(Default)]
            #[rustc_insignificant_dtor]
            pub struct Client;
            impl Drop for Client {
                fn drop(&mut self) {}
            }
        };
    );
}
