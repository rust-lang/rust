//@no-rustfix
#![warn(clippy::format_push_string)]

mod issue9493 {
    pub fn u8vec_to_hex(vector: &Vec<u8>, upper: bool) -> String {
        let mut hex = String::with_capacity(vector.len() * 2);
        for byte in vector {
            hex += &(if upper {
                format!("{byte:02X}")
                //~^ format_push_string
            } else {
                format!("{byte:02x}")
            });
        }
        hex
    }

    pub fn other_cases() {
        let mut s = String::new();
        // if let
        s += &(if let Some(_a) = Some(1234) {
            format!("{}", 1234)
            //~^ format_push_string
        } else {
            format!("{}", 1234)
        });
        // match
        s += &(match Some(1234) {
            Some(_) => format!("{}", 1234),
            //~^ format_push_string
            None => format!("{}", 1234),
        });
    }
}

mod import_write {
    mod push_str {
        // TODO: suggest importing `std::fmt::Write`;
        mod not_imported {
            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing the first time, but not again
        mod not_imported_and_not_imported {
            fn foo(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing the first time, but not again
        mod not_imported_and_imported {
            fn foo(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing, but only for `bar`
        mod imported_and_not_imported {
            fn foo(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }
    }

    mod add_assign {
        // TODO: suggest importing `std::fmt::Write`;
        mod not_imported {
            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing the first time, but not again
        mod not_imported_and_not_imported {
            fn foo(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing the first time, but not again
        mod not_imported_and_imported {
            fn foo(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        // TODO: suggest importing, but only for `bar`
        mod imported_and_not_imported {
            fn foo(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }
    }
}

fn main() {}
