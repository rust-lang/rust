//@ build-pass
#![feature(coroutines, negative_impls, stmt_expr_attributes)]
#![allow(dropping_references, dropping_copy_types)]

macro_rules! type_combinations {
    (
        $( $name:ident => { $( $tt:tt )* } );*
    ) => { $(
        mod $name {
            pub mod unsync {
                $( $tt )*

                impl !Sync for Client {}
            }
            pub mod unsend {
                $( $tt )*

                impl !Send for Client {}
            }
        }

        // This is the same bug as issue 57017, but using yield instead of await
        {
            let g = #[coroutine] move || match drop(&$name::unsync::Client::default()) {
                _status => yield,
            };
            assert_send(g);
        }

        // This tests that `Client` is properly considered to be dropped after moving it into the
        // function.
        {
            let g = #[coroutine] move || match drop($name::unsend::Client::default()) {
                _status => yield,
            };
            assert_send(g);
        }
    )* }
}

fn assert_send<T: Send>(_thing: T) {}

fn main() {
    type_combinations!(
        copy => { #[derive(Copy, Clone, Default)] pub struct Client; };
        derived_drop => { #[derive(Default)] pub struct Client { pub nickname: String } };
        significant_drop => {
            #[derive(Default)]
            pub struct Client;
            impl Drop for Client {
                fn drop(&mut self) {}
            }
        }
    );
}
