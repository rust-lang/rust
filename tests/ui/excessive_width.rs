#![allow(unused)]
#![allow(clippy::identity_op)]
#![warn(clippy::excessive_width)]

#[rustfmt::skip]
fn main() {
    let x = 1;

    let really_long_binding_name_because_this_needs_to_be_over_90_characters_long = 1usize * 200 / 2 * 500 / 1;

    {
        {
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            {
                                                {
                                                    {
                                                        {
                                                            {
                                                                {
                                                                    println!("highly indented lines do not cause a warning (by default)!")
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
