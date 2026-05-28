// rustfmt-wrap_comments: true

pub mod a_long_name {
    pub mod b_long_name {
        pub mod c_long_name {
            pub mod d_long_name {
                pub mod e_long_name {
                    pub struct Bananas;
                    impl Bananas {
                        pub fn fantastic() {}
                    }

                    pub mod f_long_name {
                        pub struct Apples;
                    }
                }
            }
        }
    }
}

/// Check out [my other struct] ([`Bananas`]) and [the method it has].
///
/// [my other struct]: a_long_name::b_long_name::c_long_name::d_long_name::e_long_name::f_long_name::Apples
/// [`Bananas`]: a_long_name::b_long_name::c_long_name::d_long_name::e_long_name::Bananas::fantastic()
/// [the method it has]: a_long_name::b_long_name::c_long_name::d_long_name::e_long_name::Bananas::fantastic()
pub struct A;
