use crate::debuggers::{extract_gdb_version, extract_lldb_version};
use crate::is_test;

#[test]
fn test_extract_gdb_version() {
    macro_rules! test { ($($expectation:literal: $input:literal,)*) => {{$(
        assert_eq!(extract_gdb_version($input), Some($expectation));
    )*}}}

    test! {
        7000001: "GNU gdb (GDB) CentOS 7.0.1-45.el5.centos",

        7002000: "GNU gdb (GDB) Red Hat Enterprise Linux 7.2-90.el6",

        7004000: "GNU gdb (Ubuntu/Linaro 7.4-2012.04-0ubuntu2.1) 7.4-2012.04",
        7004001: "GNU gdb (GDB) 7.4.1-debian",

        7006001: "GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-80.el7",

        7007001: "GNU gdb (Ubuntu 7.7.1-0ubuntu5~14.04.2) 7.7.1",
        7007001: "GNU gdb (Debian 7.7.1+dfsg-5) 7.7.1",
        7007001: "GNU gdb (GDB) Fedora 7.7.1-21.fc20",

        7008000: "GNU gdb (GDB; openSUSE 13.2) 7.8",
        7009001: "GNU gdb (GDB) Fedora 7.9.1-20.fc22",
        7010001: "GNU gdb (GDB) Fedora 7.10.1-31.fc23",

        7011000: "GNU gdb (Ubuntu 7.11-0ubuntu1) 7.11",
        7011001: "GNU gdb (Ubuntu 7.11.1-0ubuntu1~16.04) 7.11.1",
        7011001: "GNU gdb (Debian 7.11.1-2) 7.11.1",
        7011001: "GNU gdb (GDB) Fedora 7.11.1-86.fc24",
        7011001: "GNU gdb (GDB; openSUSE Leap 42.1) 7.11.1",
        7011001: "GNU gdb (GDB; openSUSE Tumbleweed) 7.11.1",

        7011090: "7.11.90",
        7011090: "GNU gdb (Ubuntu 7.11.90.20161005-0ubuntu1) 7.11.90.20161005-git",

        7012000: "7.12",
        7012000: "GNU gdb (GDB) 7.12",
        7012000: "GNU gdb (GDB) 7.12.20161027-git",
        7012050: "GNU gdb (GDB) 7.12.50.20161027-git",

        9002000: "GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2",
        10001000: "GNU gdb (GDB) 10.1 [GDB v10.1 for FreeBSD]",
    }
}

#[test]
fn test_extract_lldb_version() {
    // Apple variants
    assert_eq!(extract_lldb_version("LLDB-179.5"), Some(179));
    assert_eq!(extract_lldb_version("lldb-300.2.51"), Some(300));

    // Upstream versions
    assert_eq!(extract_lldb_version("lldb version 6.0.1"), Some(600));
    assert_eq!(extract_lldb_version("lldb version 9.0.0"), Some(900));
}

#[test]
fn is_test_test() {
    assert!(is_test("a_test.rs"));
    assert!(!is_test(".a_test.rs"));
    assert!(!is_test("a_cat.gif"));
    assert!(!is_test("#a_dog_gif"));
    assert!(!is_test("~a_temp_file"));
}

#[test]
fn string_enums() {
    // These imports are needed for the macro-generated code
    use std::fmt;
    use std::str::FromStr;

    crate::common::string_enum! {
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Animal {
            Cat => "meow",
            Dog => "woof",
        }
    }

    // General assertions, mostly to silence the dead code warnings
    assert_eq!(Animal::VARIANTS.len(), 2);
    assert_eq!(Animal::STR_VARIANTS.len(), 2);

    // Correct string conversions
    assert_eq!(Animal::Cat, "meow".parse().unwrap());
    assert_eq!(Animal::Dog, "woof".parse().unwrap());

    // Invalid conversions
    let animal = "nya".parse::<Animal>();
    assert_eq!("unknown `Animal` variant: `nya`", animal.unwrap_err());
}
