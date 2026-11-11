#![warn(clippy::format_push_string)]

fn main() {
    use std::fmt::Write;

    let mut string = String::new();
    string += &format!("{:?}", 1234);
    //~^ format_push_string

    string.push_str(&format!("{:?}", 5678));
    //~^ format_push_string

    macro_rules! string {
        () => {
            String::new()
        };
    }
    string!().push_str(&format!("{:?}", 5678));
    //~^ format_push_string
}

// TODO: recognize the already imported `fmt::Write`, and don't add a note suggesting to import it
// again
mod import_write {
    mod push_str {
        mod imported_anonymously {
            fn main(string: &mut String) {
                use std::fmt::Write as _;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported {
            fn main(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_anonymously_in_module {
            use std::fmt::Write as _;

            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_in_module {
            use std::fmt::Write;

            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_and_imported {
            fn foo(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }
    }

    mod add_assign {
        mod imported_anonymously {
            fn main(string: &mut String) {
                use std::fmt::Write as _;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported {
            fn main(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_anonymously_in_module {
            use std::fmt::Write as _;

            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_in_module {
            use std::fmt::Write;

            fn main(string: &mut String) {
                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }

        mod imported_and_imported {
            fn foo(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }

            fn bar(string: &mut String) {
                use std::fmt::Write;

                string.push_str(&format!("{:?}", 1234));
                //~^ format_push_string
            }
        }
    }
}
