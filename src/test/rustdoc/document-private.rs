// compile-flags: --document-private-items
#![crate_name = "foo"]

pub mod outmost_mod {
    pub mod outer_mod {
        pub mod inner_mod {
            /// This function is visible within `outer_mod`
            pub(in crate::outmost_mod::outer_mod) fn outer_mod_visible_fn() {}

            /// This function is visible to the entire crate
            pub(crate) fn crate_visible_fn() {}

            /// This function is visible within `outer_mod`
            pub(super) fn super_mod_visible_fn() {
                /// This function is visible since we're in the same `mod`
                inner_mod_visible_fn();
            }

            /// This function is visible only within `inner_mod`,
            /// which is the same as leaving it private.
            pub(self) fn inner_mod_visible_fn() {}

            pub mod inmost_mod {
                // @has 'foo/outmost_mod/outer_mod/inner_mod/inmost_mod/struct.Foo.html'
                // @count   - '//*[@class="visibility"]' 7
                pub struct Boo {
                    /// rhubarb rhubarb rhubarb
                    // @matches - '//*[@class="visibility"]' 'Visibility: private'
                    alpha: usize,
                    /// durian durian durian
                    pub beta: usize,
                    /// sasquatch sasquatch sasquatch
                    pub(crate) gamma: usize,
                }

                impl Boo {
                    /// Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
                    /// doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore
                    /// veritatis et quasi architecto
                    pub fn new() -> Foo {
                        Foo(0)
                    }

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    ///
                    /// # Examples:
                    ///
                    ///
                    /// ```no_run
                    /// not_pub()
                    /// ```
                    #[deprecated(since = "0.2.1", note = "The rust_foo version is more advanced")]
                    fn not_pub() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(crate) fn pub_crate() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(super) fn pub_super() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(self) fn pub_self() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(in crate::outmost_mod::outer_mod) fn pub_inner_mod() {}
                }

                pub struct Foo(
                    /// rhubarb rhubarb rhubarb
                    usize,
                    /// durian durian durian
                    pub usize,
                    /// sasquatch sasqutch sasquatch
                    pub(crate) usize,
                );

                impl Foo {
                    /// Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
                    /// doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore
                    /// veritatis et quasi architecto
                    pub fn new() -> Foo {
                        Foo(0)
                    }

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    ///
                    /// # Examples:
                    ///
                    ///
                    /// ```no_run
                    /// not_pub()
                    /// ```
                    #[deprecated(since = "0.2.1", note = "The rust_foo version is more advanced")]
                    fn not_pub() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(crate) fn pub_crate() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(super) fn pub_super() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(self) fn pub_self() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(in crate::outmost_mod::outer_mod) fn pub_inner_mod() {}
                }

                /// ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur?
                /// Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam
                /// nihil molestiae consequatur,
                pub enum Baz {
                    Size(usize),
                }

                pub enum Bar {
                    /// ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi
                    /// Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam
                    /// nihil molestiae consequatur,
                    Fizz,
                    /// ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi
                    /// Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam
                    /// nihil molestiae consequatur,
                    Pop,
                    /// ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi
                    /// Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam
                    /// nihil molestiae consequatur,
                    Bang,
                }

                impl Bar {
                    /// Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
                    /// doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore
                    /// veritatis et quasi architecto
                    pub fn new() -> Bar {
                        Fizz
                    }

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    ///
                    /// # Examples:
                    ///
                    ///
                    /// ```no_run
                    /// not_pub()
                    /// ```
                    #[deprecated(since = "0.2.1", note = "The rust_foo version is more advanced")]
                    fn not_pub() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(crate) fn pub_crate() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(super) fn pub_super() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(self) fn pub_self() {}

                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    pub(in crate::outmost_mod::outer_mod) fn pub_inner_mod() {}
                }

                pub trait Zepp {
                    fn required_method();

                    fn optional_method() {}
                }

                /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia voluptas
                /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                /// qui ratione voluptatem sequi
                pub struct Zeppish;

                impl Zepp for Zeppish {
                    /// beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia
                    /// sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos,
                    /// qui ratione voluptatem sequi
                    fn required_method() {}
                }
            }
        }
    }
}
